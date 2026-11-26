"""Step 2.3: builds data\\cases\\test_cases.jsonl, the 103 questions, from the FinanceBench question file.

From the project folder:
    python -m evals.build_cases

It can be run again at any time and gives the same file. It keeps the questions about our 37 PDFs,
moves the page numbers so that they start at 1, cleans the reasoning type, and gives every question a split:
    practice   the first 10 in file order
    test       75 of the other 93
    holdout    18 of the other 93, opened once, in the last run
The split uses a fixed seed. It keeps the same mix of question types and reasoning types in test and in holdout.
The question text belongs to FinanceBench (CC BY NC 4.0), so the folder gets its own .gitignore.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import config
from evals import cases
from ingest.names import parse_name

SPLIT_SEED = 2026  # frozen: another number gives another split
PRACTICE_COUNT = cases.EXPECTED_COUNTS["practice"]
HOLDOUT_COUNT = cases.EXPECTED_COUNTS["holdout"]
NOT_LABELLED = "Not labelled"


def read_questions(path=None, names=None):
    """The questions about our documents, in file order, as they are in the question file."""
    path = Path(path) if path else config.QUESTIONS_FILE
    if not path.exists():
        raise cases.CaseError(f"{path} is missing. Run:  python scripts\\download_pdfs.py")
    names = cases.read_names() if names is None else set(names)
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item["doc_name"] in names:
                questions.append(item)
    return questions


def clean_reasoning(raw):
    """The question file writes the reasoning type in many ways: some questions have none and some list two or three.
    This keeps the first one listed, as one of three clear names. No label at all gives Not labelled."""
    if not raw or not raw.strip():
        return NOT_LABELLED
    for part in raw.split(" OR "):
        text = part.strip().lower()
        if text.startswith("information extraction"):
            return "Information extraction"
        if text.startswith("numerical reasoning"):
            return "Numerical reasoning"
        if text.startswith("logical reasoning"):
            return "Logical reasoning"
    raise ValueError(f"Unknown reasoning type '{raw}'. Add it to clean_reasoning in evals/build_cases.py.")


# ---------- the split ----------

def seed_key(question_id):
    """A fixed, even looking number for each question. The same on every computer and every Python version."""
    return hashlib.sha256(f"{SPLIT_SEED}:{question_id}".encode("utf-8")).hexdigest()


def stratum(question):
    """The group a question belongs to for the split: its question type and its cleaned reasoning type."""
    return (question["question_type"], clean_reasoning(question.get("question_reasoning")))


def holdout_quotas(group_counts, total=None):
    """How many holdout questions each group gives: its share of the total, rounded so the sum is exact.
    The groups with the biggest left over share get the extra ones. A tie goes to the group that comes first."""
    total = HOLDOUT_COUNT if total is None else total
    size = sum(group_counts.values())
    quotas, left_over = {}, {}
    for group, count in group_counts.items():
        quotas[group], left_over[group] = divmod(total * count, size)
    extra = total - sum(quotas.values())
    for group in sorted(left_over, key=lambda name: (-left_over[name], name))[:extra]:
        quotas[group] += 1
    return quotas


def assign_splits(questions, practice=None, holdout=None):
    """The split of each question, in the same order as the questions.

    The first questions in file order are practice. The others are test or holdout: inside each group (question type
    and reasoning type) the questions are put in the order of seed_key, and the first ones go to the holdout.
    Moving or adding questions does not change who is in the holdout.
    """
    practice = PRACTICE_COUNT if practice is None else practice
    holdout = HOLDOUT_COUNT if holdout is None else holdout
    rest = questions[practice:]
    quotas = holdout_quotas(Counter(stratum(item) for item in rest), holdout)
    in_holdout = set()
    for group, quota in quotas.items():
        members = sorted((item for item in rest if stratum(item) == group), key=lambda item: seed_key(item["financebench_id"]))
        in_holdout.update(item["financebench_id"] for item in members[:quota])
    splits = []
    for index, item in enumerate(questions):
        if index < practice:
            splits.append("practice")
        else:
            splits.append("holdout" if item["financebench_id"] in in_holdout else "test")
    return splits


# ---------- one line of the file ----------

def make_case(question, split):
    """One line of the cases file, as a dict."""
    name = question["financebench_id"]
    doc = question["doc_name"]
    info = parse_name(doc)
    if question["company"] != info.company:
        raise ValueError(f"{name}: the question file says company '{question['company']}', ingest/names.py says '{info.company}'.")
    evidence = []
    for item in question["evidence"]:
        number = item.get("evidence_page_num")
        if item.get("doc_name") != doc or not isinstance(number, int):
            raise ValueError(f"{name}: an evidence item has no page number or comes from another document.")
        evidence.append({"page": number + config.PAGE_OFFSET, "text": (item.get("evidence_text") or "").strip()})
    return {
        "id": name,
        "split": split,
        "company": info.company,
        "doc": doc,
        "doc_type": info.type,
        "question_type": question["question_type"],
        "reasoning_type": clean_reasoning(question.get("question_reasoning")),
        "question": question["question"].strip(),
        "answer": question["answer"].strip(),
        "justification": (question.get("justification") or "").strip(),
        "evidence": evidence,
    }


def build_lines(questions, practice=None, holdout=None):
    return [make_case(question, split) for question, split in zip(questions, assign_splits(questions, practice, holdout))]


def check_pages(lines):
    """Is every evidence page inside its PDF? Returns (problems, number of PDFs checked), or None when the PDFs are not there."""
    import pymupdf

    page_counts, problems = {}, []
    for line in lines:
        doc = line["doc"]
        if doc not in page_counts:
            path = config.PDF_DIR / f"{doc}.pdf"
            if not path.exists():
                return None
            with pymupdf.open(path) as pdf:
                page_counts[doc] = pdf.page_count
        for item in line["evidence"]:
            if not 1 <= item["page"] <= page_counts[doc]:
                problems.append(f"{line['id']}: page {item['page']} is outside {doc}, which has {page_counts[doc]} pages")
    return problems, len(page_counts)


def write_lines(lines, path=None):
    """Write the file in one go, and put a .gitignore in its folder so Git never sees the question text."""
    path = Path(path) if path else cases.cases_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / ".gitignore").write_bytes(b"# Question text from FinanceBench (CC BY NC 4.0). Never push it.\n*\n")
    text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    temp = path.with_suffix(".tmp")
    temp.write_bytes(text.encode("utf-8"))  # bytes, so the file is the same on Windows and on Linux
    os.replace(temp, path)  # the file is never half written
    return path


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    argparse.ArgumentParser(description="Build the test set file from the question file (step 2.3).").parse_args(argv)
    try:
        questions = read_questions()
        lines = build_lines(questions)
        checked = check_pages(lines)
        print(f"{len(questions)} questions point to our {len(cases.read_names())} PDFs.")
        if checked is None:
            print(f"Pages not checked: the PDFs are not in {config.PDF_DIR}.")
        elif checked[0]:
            sys.exit("Nothing written. A page is outside its PDF:\n  " + "\n  ".join(checked[0]))
        else:
            print(f"Pages: every evidence page is inside its PDF ({checked[1]} PDFs checked).")
        path = write_lines(lines)
        print(f"Wrote {len(lines)} cases to {path}")
        loaded = cases.load_cases()  # read it back with the same loader the runs use
    except (cases.CaseError, ValueError) as err:
        sys.exit(str(err))
    print()
    print("\n".join(cases.summary_lines(loaded)))
    problems = cases.count_problems(loaded)
    if problems:
        sys.exit("PROBLEM: " + "; ".join(problems))
    print(f"\nAll {len(loaded)} lines pass the checks and the counts match the plan.")
    print("The folder has its own .gitignore, so Git never sees the questions.")
    print("Next:  python -m evals.cases --read practice")


if __name__ == "__main__":
    main()
