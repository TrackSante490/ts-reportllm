import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


def load_patient(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SYSTEM_INSTRUCTIONS = """You are a clinical documentation assistant.
Write succinct, professional reports for physicians.
Tone: objective, neutral, and clear. Format with short sections and bullet points where helpful.
Do not diagnose or prescribe. Instead, describe findings and suggest appropriate next steps or differential *as considerations*, not firm conclusions.
Do not include any advice to patients—this is clinician-facing.
Include only information provided or logically implied (no fabrication).
Use Markdown only (no code fences).
Use ATX headings only (##), and keep the exact section names provided in the user instructions.
Use "N/A" for missing fields instead of skipping sections.
Treat "vitals" and "environment" as the current readings when present.
Use "sensor_summary" and "measurement_runs" only for baseline/trend context.
Do not add chart placeholders or chart descriptions; frontend appends charts separately.
"""

USER_TEMPLATE = """Create a clinician-facing report from the following structured patient data.

# Patient Data
{patient_json}

# Output Requirements
- Use this section order:
  1) Patient Info
  2) Chief Complaint
  3) History of Present Illness
  4) Relevant History
  5) Medications & Allergies
  6) Vitals
  7) Physical Exam
  8) Environmental Assessment (if humidity or CO2 data present)
  9) Impression (concise, non-diagnostic; list key positives/negatives)
  10) Next Steps (options, not directives)
- Use exact level-2 headings (`##`) for section titles listed above.
- Keep it to ~220–380 words unless input is very long.
- For any missing or empty field, write "N/A" in that section rather than skipping it.
- Keep wording concise and avoid repeating the raw payload.

## Environmental Assessment Guidance
If humidity or CO2 data are present, include this section.
The patient payload may expose these values under an "environment" object and/or within "sensor_summary".
Prefer the latest values from "environment" when available:
- "ambient_temperature_c"
- "humidity_percent"
- "co2_ppm"
- Use those values exactly as the current environment readings; do not substitute older values from history.
- Report humidity and CO2 levels as measured
- Assess environmental conditions:
  * Humidity >80%: "Critical humidity - excellent conditions for mold growth"
  * Humidity 70-80%: "High humidity - favorable for mold growth"
  * Humidity 60-70%: "Elevated humidity - monitor for mold risk"
  * CO2 >1200 ppm: "Poor ventilation - stagnant air creates mold risk"
  * CO2 1000-1200 ppm: "Elevated CO2 - consider improving ventilation"
- Do not describe CO2 below 1000 ppm as elevated or poor ventilation.
- Do not describe humidity below 60% as elevated or as a mold-growth concern.
- If conditions favor mold growth, suggest: improve ventilation (open windows, use fans), use dehumidifier, check for visible mold
- If conditions are normal, note: "Environmental conditions appear favorable."

## Vitals Section Guidance
The patient data includes:
- "vitals": current measurements
- "sensor_summary": min/max ranges from all recent events (baseline)
- "measurement_runs": list of past measurement sessions with timestamps and vitals

Always treat "vitals" as the latest current reading.
Do not use "sensor_summary" or "measurement_runs" as the current reading if "vitals" is present.

For the Vitals section, write current values AND compare to the baseline. Example format:
"Current measurements show heart rate of 104 bpm (baseline range 70–85 bpm), temperature of 37.9°C (baseline 36.5–37.2°C), SpO2 of 97% (baseline 97–99%). Heart rate is higher than baseline."

- For any missing or empty field, write "N/A" in that section (e.g., "**Chief Complaint:** N/A").
Date of report: {today}.
"""

SECTION_ORDER_BASE = (
    "Patient Info",
    "Chief Complaint",
    "History of Present Illness",
    "Relevant History",
    "Medications & Allergies",
    "Vitals",
    "Physical Exam",
    "Impression",
    "Next Steps",
)

ENVIRONMENT_SECTION = "Environmental Assessment"


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return len(value.strip()) == 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _coalesce(*values: Any) -> Any:
    for value in values:
        if not _is_blank(value):
            return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("%", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _format_number(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return str(value).strip()
    rounded = round(number, 2)
    if float(rounded).is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _format_value(value: Any) -> str:
    if _is_blank(value):
        return "N/A"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "N/A"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, (list, tuple, set)):
        items = [_format_value(item) for item in value if not _is_blank(item)]
        return ", ".join(items) if items else "N/A"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip() or "N/A"


def _heading_key(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", title.lower())).strip()


def _has_environment_data(data: dict[str, Any]) -> bool:
    environment = data.get("environment")
    if isinstance(environment, dict):
        if any(value is not None for value in environment.values()):
            return True

    sensor_summary = data.get("sensor_summary")
    if not isinstance(sensor_summary, dict):
        return False

    for key in ("ambient_temperature_c", "humidity_percent", "co2_ppm"):
        metric = sensor_summary.get(key)
        if isinstance(metric, dict):
            if any(metric.get(field) is not None for field in ("latest", "min", "max")):
                return True
    return False


def _section_order(include_environment: bool) -> tuple[str, ...]:
    if not include_environment:
        return SECTION_ORDER_BASE
    return (
        "Patient Info",
        "Chief Complaint",
        "History of Present Illness",
        "Relevant History",
        "Medications & Allergies",
        "Vitals",
        "Physical Exam",
        ENVIRONMENT_SECTION,
        "Impression",
        "Next Steps",
    )


def _map_heading(title: str, include_environment: bool) -> str | None:
    key = _heading_key(title)
    if not key:
        return None

    if "patient info" in key or "patient information" in key:
        return "Patient Info"
    if "chief complaint" in key:
        return "Chief Complaint"
    if "history of present illness" in key or key == "hpi":
        return "History of Present Illness"
    if "relevant history" in key or "past medical history" in key or "family history" in key or "social history" in key:
        return "Relevant History"
    if "medication" in key or "allerg" in key:
        return "Medications & Allergies"
    if "vital" in key:
        return "Vitals"
    if "physical exam" in key or key == "exam":
        return "Physical Exam"
    if "environment" in key:
        return ENVIRONMENT_SECTION if include_environment else None
    if "impression" in key or key == "assessment":
        return "Impression"
    if "next steps" in key or key == "plan" or "recommend" in key:
        return "Next Steps"
    return None


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_sections(markdown: str, include_environment: bool) -> dict[str, str]:
    text = _strip_fences(markdown or "")
    if not text:
        return {}

    sections: dict[str, list[str]] = {}
    current: str | None = None

    heading_re = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
    setext_re = re.compile(r"^\s*(=+|-+)\s*$")
    bold_heading_re = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        heading_text: str | None = None

        m = heading_re.match(line)
        if m:
            heading_text = m.group(1).strip()
        else:
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            bold_match = bold_heading_re.match(stripped)
            if bold_match and setext_re.match(next_line):
                heading_text = bold_match.group(1).strip()
                i += 1
            elif stripped and setext_re.match(next_line):
                heading_text = stripped.strip("*").strip()
                i += 1

        if heading_text is not None:
            mapped = _map_heading(heading_text, include_environment)
            current = mapped
            i += 1
            continue

        if current is not None:
            sections.setdefault(current, []).append(line.rstrip())
        i += 1

    finalized: dict[str, str] = {}
    for section, chunk_lines in sections.items():
        content = "\n".join(chunk_lines).strip()
        if content:
            finalized[section] = content
    return finalized


def _extract_metric_value(values: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in values and not _is_blank(values.get(key)):
            return values.get(key)
    return None


def _format_baseline(sensor_summary: dict[str, Any], key: str) -> str | None:
    metric = sensor_summary.get(key)
    if not isinstance(metric, dict):
        return None
    low = metric.get("min")
    high = metric.get("max")
    if low is None and high is None:
        return None
    if low is not None and high is not None:
        return f"{_format_number(low)}–{_format_number(high)}"
    if low is not None:
        return f"≥ {_format_number(low)}"
    return f"≤ {_format_number(high)}"


def _build_vitals_fallback(data: dict[str, Any]) -> str:
    vitals = data.get("vitals") if isinstance(data.get("vitals"), dict) else {}
    sensor_summary = data.get("sensor_summary") if isinstance(data.get("sensor_summary"), dict) else {}

    specs = (
        ("hr", ("hr", "heart_rate", "pulse", "heart_rate_bpm"), "Heart rate", "bpm"),
        ("spo2", ("spo2", "oxygen_saturation", "spo2_percent"), "SpO2", "%"),
        ("temp", ("temp", "temperature", "temperature_c", "body_temperature_c"), "Temperature", "°C"),
        ("rr", ("rr", "resp_rate", "respiratory_rate"), "Respiratory rate", "breaths/min"),
        ("bp", ("bp", "blood_pressure"), "Blood pressure", ""),
    )

    lines: list[str] = []
    used_keys: set[str] = set()

    def _append_unit(value_text: str, unit: str) -> str:
        if not unit or value_text == "N/A":
            return value_text
        lowered = value_text.lower()
        if unit == "%" and "%" in value_text:
            return value_text
        if unit == "°C" and ("°c" in lowered or lowered.endswith("c") or " c" in lowered):
            return value_text
        if unit == "bpm" and "bpm" in lowered:
            return value_text
        if unit == "breaths/min" and ("breath" in lowered or "/min" in lowered):
            return value_text
        return f"{value_text} {unit}".replace(" %", "%")

    for summary_key, aliases, label, unit in specs:
        current = _extract_metric_value(vitals, aliases)
        if current is None:
            continue
        used_keys.update(aliases)
        current_text = _format_value(current)
        current_text = _append_unit(current_text, unit)
        baseline = _format_baseline(sensor_summary, summary_key)
        if baseline:
            lines.append(f"- {label}: {current_text} (baseline {baseline})")
        else:
            lines.append(f"- {label}: {current_text}")

    for key, value in vitals.items():
        if key in used_keys or _is_blank(value):
            continue
        label = key.replace("_", " ").title()
        lines.append(f"- {label}: {_format_value(value)}")

    return "\n".join(lines) if lines else "N/A"


def _environment_value(data: dict[str, Any], key: str) -> Any:
    environment = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    if key in environment and environment.get(key) is not None:
        return environment.get(key)

    sensor_summary = data.get("sensor_summary") if isinstance(data.get("sensor_summary"), dict) else {}
    metric = sensor_summary.get(key)
    if not isinstance(metric, dict):
        return None
    if metric.get("latest") is not None:
        return metric.get("latest")
    if metric.get("max") is not None:
        return metric.get("max")
    if metric.get("min") is not None:
        return metric.get("min")
    return None


def _build_environment_fallback(data: dict[str, Any]) -> str:
    ambient_temp = _environment_value(data, "ambient_temperature_c")
    humidity = _environment_value(data, "humidity_percent")
    co2 = _environment_value(data, "co2_ppm")

    lines: list[str] = []
    if ambient_temp is not None:
        lines.append(f"- Ambient temperature: {_format_number(ambient_temp)}°C")
    if humidity is not None:
        lines.append(f"- Humidity: {_format_number(humidity)}%")
    if co2 is not None:
        lines.append(f"- CO2: {_format_number(co2)} ppm")

    humidity_num = _as_float(humidity)
    co2_num = _as_float(co2)
    if humidity_num is not None or co2_num is not None:
        if (humidity_num is not None and humidity_num > 70) or (co2_num is not None and co2_num > 1000):
            lines.append("- Environmental factors may be contributing context and are worth clinical awareness.")
        else:
            lines.append("- Environmental conditions appear favorable.")

    return "\n".join(lines) if lines else "N/A"


def _build_impression_fallback(data: dict[str, Any], include_environment: bool) -> str:
    points: list[str] = []
    complaint = _format_value(data.get("chief_complaint"))
    exam = _format_value(data.get("exam"))

    def _as_sentence(text: str) -> str:
        trimmed = text.strip()
        if not trimmed:
            return trimmed
        if trimmed.endswith((".", "!", "?")):
            return trimmed
        return f"{trimmed}."

    if complaint != "N/A" and complaint.lower() not in {"none", "no complaint"}:
        points.append(f"Chief concern: {_as_sentence(complaint)}")
    if exam != "N/A":
        points.append(f"Exam findings: {_as_sentence(exam)}")
    if isinstance(data.get("vitals"), dict) and data.get("vitals"):
        points.append("Current vitals reviewed with baseline trend context.")
    if include_environment:
        points.append("Environmental readings considered for clinician awareness.")

    if not points:
        return "N/A"
    return "\n".join(f"- {point}" for point in points)


def _build_next_steps_fallback(data: dict[str, Any], include_environment: bool) -> str:
    options: list[str] = []
    if _format_value(data.get("exam")) != "N/A":
        options.append("Consider targeted follow-up of physical exam findings with appropriate specialty input.")
    if isinstance(data.get("vitals"), dict) and data.get("vitals"):
        options.append("Consider repeat measurements and trend comparison with prior sessions.")
    if include_environment:
        options.append("Consider ventilation and humidity mitigation if elevated readings persist.")
    options.append("Correlate findings with history, medications, and allergies before care-plan decisions.")
    return "\n".join(f"- {option}" for option in options[:4])


def _build_fallback_sections(data: dict[str, Any], include_environment: bool) -> dict[str, str]:
    patient_lines = [
        f"- Patient ID: {_format_value(data.get('patient_id'))}",
        f"- Name: {_format_value(data.get('name'))}",
        f"- Age: {_format_value(data.get('age'))}",
        f"- Sex: {_format_value(data.get('sex'))}",
    ]

    relevant_history = [
        f"- Past Medical History: {_format_value(data.get('past_medical_history'))}",
        f"- Family History: {_format_value(data.get('family_history'))}",
        f"- Social History: {_format_value(data.get('social_history'))}",
    ]

    medication_lines = [
        f"- Medications: {_format_value(data.get('medications'))}",
        f"- Allergies: {_format_value(data.get('allergies'))}",
    ]

    exam_lines = [f"- Findings: {_format_value(data.get('exam'))}"]
    attachments = data.get("attachments")
    if isinstance(attachments, (list, tuple)) and attachments:
        attachment_items = [str(item).strip() for item in attachments if str(item).strip()]
        if attachment_items:
            exam_lines.append("- Attachments: " + ", ".join(attachment_items[:6]))

    fallback = {
        "Patient Info": "\n".join(patient_lines),
        "Chief Complaint": _format_value(data.get("chief_complaint")),
        "History of Present Illness": _format_value(data.get("history_of_present_illness")),
        "Relevant History": "\n".join(relevant_history),
        "Medications & Allergies": "\n".join(medication_lines),
        "Vitals": _build_vitals_fallback(data),
        "Physical Exam": "\n".join(exam_lines),
        "Impression": _build_impression_fallback(data, include_environment),
        "Next Steps": _build_next_steps_fallback(data, include_environment),
    }
    if include_environment:
        fallback[ENVIRONMENT_SECTION] = _build_environment_fallback(data)
    return fallback


def _is_noisy(content: str) -> bool:
    text = content.strip()
    if not text:
        return True
    if len(text) > 2200:
        return True
    brace_count = text.count("{") + text.count("}")
    if brace_count >= 8 and brace_count / max(len(text), 1) > 0.02:
        return True
    return False


def _clean_section_content(content: str) -> str:
    text = content.strip()
    if not text:
        return ""
    return text.strip()


def _looks_incomplete(section: str, content: str) -> bool:
    text = content.strip()
    if not text:
        return True
    if section != "Patient Info":
        return False

    key = _heading_key(text)
    has_markers = any(marker in key for marker in ("patient id", "name", "age", "sex"))
    line_count = len([line for line in text.splitlines() if line.strip()])
    return (not has_markers) or line_count < 2


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = _coalesce(item.get("text"), item.get("content"))
                if text is not None:
                    parts.append(str(text))
                    continue
            parts.append(str(item))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def post_process_report(markdown: str, data: dict[str, Any]) -> str:
    include_environment = _has_environment_data(data)
    parsed_sections = _extract_sections(markdown, include_environment)
    fallback_sections = _build_fallback_sections(data, include_environment)

    lines = [
        "# Clinical Report",
        "",
        f"_Date: {datetime.now().strftime('%Y-%m-%d')}_",
        "",
    ]

    for section in _section_order(include_environment):
        lines.append(f"## {section}")
        lines.append("")
        parsed = _clean_section_content(parsed_sections.get(section, ""))
        if _is_noisy(parsed) or _looks_incomplete(section, parsed):
            parsed = ""
        content = parsed or fallback_sections.get(section, "N/A")
        lines.append(content if content.strip() else "N/A")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_prompt(data: dict) -> list:
    today = datetime.now().strftime("%Y-%m-%d")
    user = USER_TEMPLATE.format(
        patient_json=json.dumps(data, ensure_ascii=False, indent=2),
        today=today,
    )
    return [SystemMessage(content=SYSTEM_INSTRUCTIONS), HumanMessage(content=user)]


def generate_report(data: dict, model: ChatOllama) -> str:
    messages = build_prompt(data)
    resp = model.invoke(messages)
    raw_markdown = _flatten_content(getattr(resp, "content", ""))
    return post_process_report(raw_markdown, data)


def main():
    parser = argparse.ArgumentParser(description="Generate a medical report from patient JSON.")
    parser.add_argument("json_path", nargs="?", default="patients.json", help="Path to patient JSON file (default: patients.json)")
    parser.add_argument("-o", "--out", default="report.md", help="Output file prefix (default: report.md)")
    parser.add_argument("--models", nargs="+", default=["llama3.1", "phi3", "mistral", "qwen2.5"],
                        help="List of Ollama models to compare (space-separated)")
    args = parser.parse_args()

    data = load_patient(args.json_path)

    for model_name in args.models:
        print(f"\n Running model: {model_name} ...")
        try:
            model = ChatOllama(model=model_name, temperature=0)
            report_text = generate_report(data, model)
            out_path = Path(f"{Path(args.out).stem}_{model_name.replace(':', '-')}.md")
            out_path.write_text(report_text, encoding="utf-8")
            print(f" Report written to {out_path.resolve()}")
        except Exception as e:
            print(f" Error with {model_name}: {e}")


if __name__ == "__main__":
    main()
