import os
import json
import argparse
from pathlib import Path
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
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
If info is missing, write “Not provided”.
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
  6) Vitals
  7) Physical Exam
  8) Impression (concise, non-diagnostic; list key positives/negatives)
  9) Next Steps (options, not directives)
- Keep it to ~200–350 words unless input is very long.
- Use Markdown with headings.
- If images/attachments are listed, reference them by name but don’t invent content.

Date of report: {today}.
"""

def build_prompt(data: dict) -> list:
    today = datetime.now().strftime("%Y-%m-%d")
    user = USER_TEMPLATE.format(
        patient_json=json.dumps(data, ensure_ascii=False, indent=2),
        today=today,
    )
    return [SystemMessage(content=SYSTEM_INSTRUCTIONS), HumanMessage(content=user)]

def generate_report(data: dict, model: ChatGoogleGenerativeAI) -> str:
    messages = build_prompt(data)
    resp = model.invoke(messages)
    return resp.content

def main():
    parser = argparse.ArgumentParser(description="Generate a medical report from patient JSON.")
    parser.add_argument("json_path", help="Path to patient JSON file")
    parser.add_argument("-o", "--out", default="report.md", help="Output file (default: report.md)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Model name")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set. Please set the environment variable and try again.")

    data = load_patient(args.json_path)
    model = ChatGoogleGenerativeAI(model=args.model)

    report_text = generate_report(data, model)

    out_path = Path(args.out)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to {out_path.resolve()}")

if __name__ == "__main__":
    main()
