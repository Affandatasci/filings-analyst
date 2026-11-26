# Filings Analyst

A question answering bot over real SEC filings that shows the page it used. Work in progress.

To run it: `pip install -r requirements.txt`, copy `.env.example` to `.env` and add your keys, then run `python scripts/download_pdfs.py`, `python -m ingest.build_index` and `python scripts/check_index.py`.

Data: the filings and questions come from FinanceBench by Patronus AI (CC BY NC 4.0, for portfolio use). They are not stored in this repo.
