"""
Command-line demo for the role normalization + requirement extraction
pipeline: raw job dict -> normalized Role -> extracted RoleRequirement
objects.

Makes a real OpenAI call (requires OPENAI_API_KEY) and prints both the
normalized Role and the extracted requirements as JSON.

Usage:
    python -m backend.ingestion.demo
    python -m backend.ingestion.demo --model gpt-4o
"""

from __future__ import annotations

import argparse
import json

from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import extract_role_requirements

# Deliberately uses inconsistent source keys (jobTitle/companyName/jobUrl)
# to exercise the normalization layer, the way a real scraped or
# third-party API payload would.
SAMPLE_RAW_JOB = {
    "jobTitle": "Quantitative Research Intern",
    "companyName": "Meridian Capital",
    "jobUrl": "https://meridiancapital.example/careers/quant-intern",
    "location": "New York, NY",
    "jobDescription": (
        "Meridian Capital is looking for a Quantitative Research Intern to "
        "join our systematic trading team. You will build and backtest "
        "alpha signals using our internal Python and C++ research "
        "libraries.\n\n"
        "Requirements:\n"
        "- Strong foundation in probability and statistics\n"
        "- Proficiency in Python; C++ experience is a plus\n"
        "- Comfortable with algorithms and data structures\n"
        "- Familiarity with market microstructure or trading systems is a "
        "plus but not required\n"
        "- Coursework or project experience in machine learning is a plus"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a sample raw job dict through normalization and requirement extraction."
    )
    parser.add_argument("--model", help="Override the configured OpenAI model")
    args = parser.parse_args()

    role = normalize_role(SAMPLE_RAW_JOB)
    print("--- Normalized Role ---")
    print(role.model_dump_json(indent=2))

    requirements = extract_role_requirements(role, model=args.model)
    print("\n--- Extracted RoleRequirements ---")
    print(json.dumps([r.model_dump() for r in requirements], indent=2))


if __name__ == "__main__":
    main()
