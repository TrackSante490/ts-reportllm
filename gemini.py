import argparse
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from .report_llm import generate_report, load_patient
except ImportError:
    from report_llm import generate_report, load_patient


def main():
    parser = argparse.ArgumentParser(description="Generate a medical report from patient JSON.")
    parser.add_argument("json_path", nargs="?", default="patients.json", help="Path to patient JSON file (default: patients.json)")
    parser.add_argument("-o", "--out", default="report.md", help="Output file (default: report.md)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Model name")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set. Please set the environment variable and try again.")

    data = load_patient(args.json_path)
    model = ChatGoogleGenerativeAI(model=args.model, google_api_key=api_key)

    report_text = generate_report(data, model)

    out_path = Path(args.out)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
