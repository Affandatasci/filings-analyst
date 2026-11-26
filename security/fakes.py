"""Step 3.1: the facts about the 20 fake files, read from data\\fake_pdfs\\manifest.json.

scripts\\make_fake_pdfs.py writes the manifest next to the PDFs. The guards, the agent tools, the attack tests and the
index builder all read it here, so the fake names, the secret words and the access levels live in one place only.
Without the manifest (the fake files are not made yet) every function gives an empty answer and nothing breaks.

Keep data\\fake_pdfs\\manifest.json in Git: the app needs it to know which words are secret. It holds fake data only.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import config

COMPANY = "Northwind"  # the made up company the 20 fake files belong to
YEAR = 2024


def pdf_dir():
    return Path(config.DATA_DIR) / "fake_pdfs"


def manifest_path():
    return pdf_dir() / "manifest.json"


@lru_cache(maxsize=8)
def _read(path, stamp):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest():
    """The manifest as a dict, or an empty one when it does not exist yet. It is read again when the file changes."""
    path = manifest_path()
    try:
        return _read(str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} is damaged ({type(err).__name__}). Make it again with:  python scripts\\make_fake_pdfs.py") from err


def files():
    """One dict per fake file: name, kind, level, type, title, pages, people, emails, phones, emp_ids, secret_words."""
    return list(load_manifest().get("files", []))


def _all(key):
    return sorted({value for item in files() for value in item.get(key, [])})


def secret_words():
    return _all("secret_words")


def known_names():
    return _all("people")


def personal_values():
    """Every fake email, phone number and employee ID, exactly as written in the files."""
    return sorted(set(_all("emails")) | set(_all("phones")) | set(_all("emp_ids")))


def canary():
    return load_manifest().get("canary", "")


def level_of(doc):
    """The access level of a fake file, or None when the name is not one of the fake files."""
    for item in files():
        if item["name"].lower() == str(doc).lower():
            return item["level"]
    return None


def catalog():
    """Name, company, year, type and level of every fake file, for the agent tool list_documents."""
    return [
        {"name": item["name"], "company": COMPANY, "year": YEAR, "type": item["type"], "level": item["level"], "title": item.get("title", "")}
        for item in files()
    ]


def fake_documents(only=None):
    """The fake files as Document objects for the index builder. Empty when there is no manifest."""
    items = files()
    if not items:
        return []
    from ingest.names import DocInfo
    from ingest.read_pdf import Document

    wanted = {name.lower() for name in only} if only else None
    documents = []
    for item in items:
        if wanted is not None and item["name"].lower() not in wanted:
            continue
        path = pdf_dir() / f"{item['name']}.pdf"
        if not path.exists():
            sys.exit(f"Missing fake PDF: {path}\nMake the fake files again with:  python scripts\\make_fake_pdfs.py")
        info = DocInfo(name=item["name"], company=COMPANY, year=YEAR, period="", type=item["type"])
        documents.append(Document(info, path, source="fake", access=item["level"]))
    return documents
