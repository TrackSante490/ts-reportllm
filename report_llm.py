import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

def load_patient(path: str) -> dict:
    with open("patients.json", "r", encoding="utf-8") as f:
        return json.load(f)

SYSTEM_INSTRUCTIONS = """You are a clinical documentation assistant.
Write succinct, professional reports for physicians.
Tone: objective, neutral, and clear. Format with short sections and bullet points where helpful.
Do not diagnose or prescribe. Instead, describe findings and suggest appropriate next steps or differential *as considerations*, not firm conclusions.
Do not include any advice to patients—this is clinician-facing.
Include only information provided or logically implied (no fabrication).
For any missing or empty fields in the patient data, write "N/A" in that section rather than skipping it.

When writing the Vitals section, always include trend analysis:
- Compare current measurements to the patient's baseline (from sensor_summary ranges).
- Review measurement_runs to identify trends (improving, stable, or worsening).
- Explicitly note if any values are elevated, low, or trending differently than baseline.
- Example: "Heart rate of 104 bpm is elevated compared to the patient's recent baseline of 70–85 bpm."

End with a brief 'Next Steps' section phrased as options to consider.
"""

USER_TEMPLATE = """Create a clinician-facing report from the following structured patient data.

# Patient Data
{patient_json}

# Output Requirements
- Use this section order:
  1) Patient Info
  2) Chief Complaint
  3) History of Present Illness
  4) Relevant History (PMH, FHx, SHx)
  5) Medications & Allergies
  6) Vitals (with trend analysis compared to baseline and recent sessions)
  7) Physical Exam
  8) Impression (concise, non-diagnostic; list key positives/negatives)
  9) Next Steps (options, not directives)
- Keep it to ~200–350 words unless input is very long.
- Use Markdown with headings.
- If images/attachments are listed, reference them by name but don’t invent content.

## Vitals Section Guidance
The patient data includes:
- "vitals": current measurements
- "sensor_summary": min/max ranges from all recent events (baseline)
- "measurement_runs": list of past measurement sessions with timestamps and vitals

For the Vitals section, write current values AND compare to the baseline. Example format:
"Current measurements show heart rate of 104 bpm (baseline range 70–85 bpm), temperature of 37.9°C (baseline 36.5–37.2°C), SpO2 of 97% (baseline 97–99%), and blood pressure of 148/94 mmHg (baseline 120–130/80–85 mmHg). Blood pressure appears persistently elevated. Heart rate is higher than baseline."

- For any missing or empty field, write "N/A" in that section (e.g., "**Chief Complaint:** N/A").
Date of report: {today}.
"""

# ## Vitals Section Guidance
# For the Vitals section, analyze the current measurements and compare to baseline:
# - Report each vital with its current value and baseline range
# - Explicitly state if each value is elevated, low, or normal compared to baseline
# - Look at measurement_runs to identify trends: are values improving, stable, or worsening?
# - Mention any notable patterns (e.g., "heart rate trending upward over recent sessions")
# - Connect findings to the chief complaint if relevant

# Examples of varied analyses:
# - "Heart rate of 104 bpm is elevated (baseline 70–85 bpm), a notable increase from yesterday's 88 bpm."
# - "Blood pressure remains persistently elevated at 148/94 mmHg despite medication history."
# - "Temperature of 37.9°C suggests possible fever; this represents a significant increase from baseline."
# - "SpO2 of 95% is slightly low compared to patient's usual 97–99% range."


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
    return resp.content

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
