"""Step 2.3: the test set, read from data\\cases\\test_cases.jsonl.

    from evals.cases import load_cases
    cases = load_cases("test")      # the 75 test cases, in file order
    cases = load_cases()            # all 103
    for case in cases:
        print(case.id, case.question, case.answer, case.pages)

The file is made by evals/build_cases.py. Every line is checked when it is read. A bad line stops
the run and names the line, so a broken file can never give a quiet wrong score.

Check it from the project folder:
    python -m evals.cases                    check the file and print the counts
    python -m evals.cases --read practice    write data\\cases\\read_practice.txt, to read in VS Code
    python -m evals.cases --read test        the same for the 75 test questions
The holdout is opened once, in the last run, so --read refuses it.
"""
import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import config

SPLITS = ("practice", "test", "holdout")
EXPECTED_COUNTS = {"practice": 10, "test": 75, "holdout": 18}  # the plan: 103 in all
REASONING_TYPES = ("Information extraction", "Numerical reasoning", "Logical reasoning", "Not labelled")
TEXT_FIELDS = ("id", "split", "company", "doc", "doc_type", "question_type", "reasoning_type", "question", "answer")


class CaseError(ValueError):
    """The cases file is missing or a line in it is bad. The message names the line and the problem."""


@dataclass(frozen=True)
class Evidence:
    page: int  # our page number, which starts at 1
    text: str  # the evidence text from the question file


@dataclass(frozen=True)
class Case:
    id: str
    split: str  # practice, test or holdout
    company: str
    doc: str  # the PDF name without .pdf, for example 3M_2018_10K
    doc_type: str  # 10K, 10Q, 8K or EARNINGS
    question_type: str  # as written in the question file
    reasoning_type: str  # one of REASONING_TYPES
    question: str
    answer: str  # the reference answer
    justification: str  # how the reference answer was found. Empty for some questions
    evidence: tuple  # of Evidence, at least one

    @property
    def pages(self):
        """The evidence as (doc, page) pairs: the same shape as the pages an answer cites."""
        return [(self.doc, item.page) for item in self.evidence]


# ---------- reading the file ----------

def cases_path():
    return config.DATA_DIR / "cases" / "test_cases.jsonl"


def read_names():
    """The document names in data\\pdf_list.txt (our 37 PDFs)."""
    if not config.PDF_LIST.exists():
        raise CaseError(f"{config.PDF_LIST} is missing. Run:  python scripts\\download_pdfs.py")
    return {line.strip() for line in config.PDF_LIST.read_text(encoding="utf-8").splitlines() if line.strip()}


def parse_case(data, names):
    """Turn one decoded line into a Case. Raises ValueError with the problem in plain words."""
    if not isinstance(data, dict):
        raise ValueError("the line is not a JSON object")
    for key in TEXT_FIELDS:
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"'{key}' is missing or empty")
    if data["split"] not in SPLITS:
        raise ValueError(f"split '{data['split']}' is not one of: {', '.join(SPLITS)}")
    if data["reasoning_type"] not in REASONING_TYPES:
        raise ValueError(f"reasoning_type '{data['reasoning_type']}' is not one of: {', '.join(REASONING_TYPES)}")
    if data["doc"] not in names:
        raise ValueError(f"unknown doc '{data['doc']}' (it is not in {config.PDF_LIST.name})")
    justification = data.get("justification", "")
    if not isinstance(justification, str):
        raise ValueError("'justification' must be text")
    evidence = data.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("no evidence page: 'evidence' must be a list with at least one item")
    items = []
    for number, item in enumerate(evidence, start=1):
        page = item.get("page") if isinstance(item, dict) else None
        text = item.get("text") if isinstance(item, dict) else None
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError(f"evidence item {number} has no valid page (pages start at 1)")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"evidence item {number} has no text")
        items.append(Evidence(page, text))
    return Case(
        data["id"], data["split"], data["company"], data["doc"], data["doc_type"], data["question_type"],
        data["reasoning_type"], data["question"], data["answer"], justification, tuple(items),
    )


def load_cases(split=None, path=None, names=None):
    """Read and check the cases file. Returns a list of Case in file order.

    split: practice, test or holdout. None gives all of them.
    Raises CaseError with the line number when the file or a line is bad.
    """
    if split is not None and split not in SPLITS:
        raise CaseError(f"Unknown split '{split}'. Use one of: {', '.join(SPLITS)}.")
    path = Path(path) if path else cases_path()
    if not path.exists():
        raise CaseError(f"{path} is missing. Run:  python -m evals.build_cases")
    names = read_names() if names is None else set(names)
    loaded, first_line = [], {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except ValueError as err:
            raise CaseError(f"{path.name} line {number}: not valid JSON ({err})") from err
        try:
            case = parse_case(data, names)
        except ValueError as err:
            raise CaseError(f"{path.name} line {number}: {err}") from err
        if case.id in first_line:
            raise CaseError(f"{path.name} line {number}: id '{case.id}' is already used on line {first_line[case.id]}")
        first_line[case.id] = number
        loaded.append(case)
    if not loaded:
        raise CaseError(f"{path} has no cases. Run:  python -m evals.build_cases")
    return [case for case in loaded if split in (None, case.split)]


def count_problems(cases):
    """Compare the counts with the plan. Returns plain sentences, and an empty list when all is right."""
    problems = []
    for split in SPLITS:
        found = sum(1 for case in cases if case.split == split)
        if found != EXPECTED_COUNTS[split]:
            problems.append(f"{split} has {found} cases, the plan says {EXPECTED_COUNTS[split]}")
    return problems


# ---------- showing it ----------

def summary_lines(cases):
    """The counts per split, then per question type, reasoning type and document type with the splits side by side."""
    counts = ", ".join(f"{split} {sum(1 for case in cases if case.split == split)}" for split in SPLITS)
    lines = [f"{len(cases)} cases: {counts}"]
    for title, key in (("Question type", "question_type"), ("Reasoning type", "reasoning_type"), ("Document type", "doc_type")):
        lines.append(f"{title} (practice / test / holdout):")
        for value in sorted({getattr(case, key) for case in cases}):
            part = [sum(1 for case in cases if getattr(case, key) == value and case.split == split) for split in SPLITS]
            lines.append(f"  {value:<24} {part[0]:>3} / {part[1]:>3} / {part[2]:>3}")
    return lines


def reading_text(cases):
    """One block per case: question, reference answer, why, and the evidence text. For reading in VS Code."""
    blocks = []
    for number, case in enumerate(cases, start=1):
        lines = [
            f"=== {number} of {len(cases)} | {case.id} | {case.split} ===",
            f"Company:   {case.company} | {case.doc_type} | {case.doc}",
            f"Type:      {case.question_type} | {case.reasoning_type}",
            f"Question:  {case.question}",
            f"Reference: {case.answer}",
        ]
        if case.justification.strip():
            lines.append(f"Why:       {' '.join(case.justification.split())}")
        for item in case.evidence:
            lines.append(f"Evidence on page {item.page}:")
            lines.append(textwrap.fill(" ".join(item.text.split()), width=100, initial_indent="   ", subsequent_indent="   "))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Check the test set file, or write a text file of questions to read (step 2.3).")
    parser.add_argument("--read", choices=SPLITS, help="write data\\cases\\read_<split>.txt: questions, reference answers and evidence")
    args = parser.parse_args(argv)
    if args.read == "holdout":
        sys.exit("The holdout is opened once, in the last run. It is not for reading now.")
    try:
        cases = load_cases()
    except CaseError as err:
        sys.exit(str(err))
    if args.read:
        chosen = [case for case in cases if case.split == args.read]
        path = cases_path().parent / f"read_{args.read}.txt"
        path.write_bytes(reading_text(chosen).encode("utf-8"))
        print(f"Wrote {len(chosen)} questions to {path}")
        print("Open it in VS Code and read it. It stays out of Git.")
        return
    print("\n".join(summary_lines(cases)))
    problems = count_problems(cases)
    if problems:
        sys.exit("PROBLEM: " + "; ".join(problems))
    print(f"All {len(cases)} lines pass the checks and the counts match the plan.")


if __name__ == "__main__":
    main()
