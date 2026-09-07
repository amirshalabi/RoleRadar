"""
Command-line demo for candidate profile extraction.

Makes a real OpenAI call (requires OPENAI_API_KEY) and prints the
resulting CandidateProfile as JSON, so extraction quality can be
eyeballed against real or sample resume text without going through
Streamlit.

Usage:
    python -m backend.candidate.demo
    python -m backend.candidate.demo --text "paste resume text here"
    python -m backend.candidate.demo --file path/to/resume.pdf
    python -m backend.candidate.demo --file path/to/resume.txt
    python -m backend.candidate.demo --model gpt-4o
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.candidate.parser import extract_text_from_pdf
from backend.candidate.profile import extract_candidate_profile

SAMPLE_RESUME = """\
Jane Doe
B.S. Computer Science, State University, Expected May 2026, GPA 3.8
Relevant coursework: Data Structures, Operating Systems, Machine Learning

Skills: Python, C++, React, Docker

Projects:
- Distributed Key-Value Store (C++): Built a Raft-based distributed
  key-value store in C++ supporting leader election and log replication,
  tested with simulated network partitions across 5 nodes.

Experience:
- Software Engineering Intern, Acme Corp, Summer 2025: Built internal
  React dashboards consumed by 50+ engineers; wrote Python ETL scripts
  scheduled via cron.
"""


def _load_resume_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        path = Path(args.file)
        if path.suffix.lower() == ".pdf":
            return extract_text_from_pdf(path)
        return path.read_text()
    return SAMPLE_RESUME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a CandidateProfile from resume text and print it as JSON."
    )
    parser.add_argument("--text", help="Raw resume text")
    parser.add_argument("--file", help="Path to a .pdf or .txt resume file")
    parser.add_argument("--model", help="Override the configured OpenAI model")
    args = parser.parse_args()

    resume_text = _load_resume_text(args)
    profile = extract_candidate_profile(resume_text, model=args.model)
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
