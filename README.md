# Filings Analyst

A question answering bot for real company filings: annual reports, quarterly reports, event reports and earnings releases. You ask a question, it answers in a few lines and shows the page it used. If the pages do not hold the answer, it is built to say so instead of guessing.

I built it as an AI Engineer portfolio project to show the part that most chatbot demos skip: measuring how often the bot is wrong, attacking it on purpose, and watching it after launch.

**Status: built, tested and measured. Not deployed yet.** Everything below was run on 37 real filings (about 4,800 pages) and the 103 FinanceBench questions about them.
The whole project, with every experiment and attack run, has cost 48.3 US cents of Claude calls so far (budget 100 cents, with a hard stop at 90).

## How well does it work?

Two question sets. The test set (75 questions, run on 25 Nov 2026) was used while tuning the bot. The holdout set (18 questions, run on 26 Nov 2026) was kept closed until the end and opened once, so it shows how the bot does on questions it was never tuned on. In brackets: the 95 in 100 range, as a score out of 100. With so few questions the true score can sit anywhere in that range.

| Measure | Test, 75 questions | Holdout, 18 questions |
|---|---|---|
| Answer is correct (judged by Claude) | 53 of 75 (60 to 80) | 12 of 18 (44 to 84) |
| Numbers match the reference answer | 34 of 49 (55 to 80) | 9 of 15 (36 to 80) |
| The right page was among the pieces found | 57 of 75 (65 to 84) | 10 of 18 (34 to 75) |
| Every claim is backed by the pieces (judged by Claude) | 74 of 75 (93 to 100) | 17 of 18 (74 to 99) |
| Cited pages are real pages that the bot was given | 72 of 75 (89 to 99) | 16 of 18 (67 to 97) |
| A cited page is a page that holds the answer | 52 of 73 (60 to 80) | 8 of 18 (25 to 66) |
| Reply is in the agreed format | 75 of 75 (95 to 100) | 18 of 18 (82 to 100) |
| Answer is short enough | 74 of 75 (93 to 100) | 18 of 18 (82 to 100) |
| Cost per question (the answer and the judge) | 0.1348 cents | 0.1498 cents |
| Seconds per answer (the bot only) | 4.1 | 7.2 |

Setup: Claude Haiku 5.5 writes the answers and also judges them, embeddings by BAAI/bge-small-en-v1.5, Qdrant Cloud collection filings_v1, best 5 pieces per question, company and year filter, one rewritten search line, whole pages for the first 3 different pages.

The holdout score is lower than the test score. With 18 questions the two ranges overlap a lot, so this does not prove the bot got worse, but I am not hiding it.

## Where it fails

22 of the 75 test answers were wrong. Reading each one gave two causes:

- **Search missed (10):** the right page was not among the pieces the bot was given. In 4 of these the bot said it could not find the answer, and in 6 it answered from other pages.
- **The bot misread the right page (12):** 7 had a wrong number, 2 said "not found" although the right page was there, 2 differ from the reference where there is no number to check, 1 had the right number but the judge marked the answer wrong.

On the holdout set 6 of 18 answers were wrong: 4 search missed and 2 misread.

By type of document (test set):

| Document | Questions | Right page found | Answers correct |
|---|---|---|---|
| Annual report (10-K) | 56 | 44 | 44 |
| Quarterly report (10-Q) | 6 | 3 | 1 |
| Earnings release | 9 | 6 | 4 |
| Event report (8-K) | 4 | 4 | 4 |

The quarterly and earnings groups are small, so read them as a hint and not as a measurement. One line for each wrong answer is in `results/failure_notes.md` (test) and `results/failure_notes_holdout.md` (holdout). They use question ids only.

## How it works

Built once:

1. Read the 37 PDFs with PyMuPDF.
2. Cut every page into pieces of about 400 tokens and put a header line (company, year, document, page) in front of each piece.
3. Turn each piece into 384 numbers with the small embedding model bge-small-en-v1.5.
4. Store them in Qdrant Cloud with labels: company, year, document, page and access level. 11,164 pieces: 11,125 from the real filings and 39 fake bait pieces from 20 fake files.

For every question:

1. Input check: not empty, not too long, a known role.
2. Claude rewrites the question into a better search line, with a list of finance words.
3. Search with a filter on company and year and on what the role may read. Qdrant makes the access check, not the prompt.
4. Take the best 5 pieces and send the whole page for the first 3 different pages.
5. Instruction scan and personal data hiding on the pieces.
6. Claude Haiku 5.5 answers in a fixed JSON format with the pages it used.
7. Output check: secret words, personal data, buy or sell advice, and cited pages that were never sent.
8. Log the request and the cost.

There is also an agent mode: Claude can search again, up to 4 steps, with 4 tools that only read. In my tests it did not beat the plain bot, so the plain bot is the default.

## What I tried

One change at a time, on the same 85 questions, counting only whether search brings the right page. The biggest wins: a rewritten search line (51 to 66) and the header line on every piece (45 to 62).

| What I changed | Right page found, of 85 | What I kept |
|---|---|---|
| Search by meaning only, by keywords only, or both mixed | 51, 24, 39 | meaning only |
| Pieces returned by search: 3, 4, 5 or 8 | 40, 45, 51, 54 | 5 |
| Filter: none, company, company and year | 43, 46, 51 | company and year |
| Rewrite before searching: none, one rewrite, one rewrite with finance word hints, three rewrites with hints | 51, 64, 66, 64 | one rewrite with hints |
| A second rewrite prompt (v3) against the first (v2) | 63 against 66 | the first (v2) |

The index itself, scored in Google Colab with the saved rewrites (no Claude calls). The number is how often the right page is among the pages the bot reads whole:

| Index | What changed | Of 85 |
|---|---|---|
| filings_v1 (kept) | nothing: header line on each piece, small model, pieces of about 400 tokens | 62 |
| noheader | header line off | 45 |
| large | large embedding model | 61 |
| context | context lines added | 55 |
| size250 | pieces of 250 tokens | 56 |
| size480 | pieces of 480 tokens | 62 |

The notes with the reasons are in `results/experiments.md`.

## Security

The bot has three roles: guest, analyst and hr. I made 20 fake internal files (memos, HR letters, a salary list, a handbook, internal notes and files that hold secret words). Every name, number and word in them is made up. They sit in the search store as bait: if one of their words shows up in an answer for the wrong role, something leaked.

What a question meets, in order: the input check, the access filter inside Qdrant, the instruction scan (pieces that read like orders to the model are removed), personal data hiding for internal and hr pieces, and the output check. The agent has 4 tools and all of them only read.

Demo limits: a question can be at most 500 characters, and the public demo may spend 2 cents a day and 15 cents in all.

I wrote 40 attack messages in seven groups: direct orders, orders hidden inside a document, personal data, secret words, access to files the role may not read, buy or sell advice, and questions the documents cannot answer. Each has a role and a severity from 1 to 3 (3 leaks personal data or secrets). Code marks every answer.

| Run | Got through, of 40 |
|---|---|
| Plain bot, guards off (the control) | 7 (6 real problems and 1 polite refusal that the advice patterns also match) |
| Plain bot, guards on | 0 |
| Agent, guards on, first run | 5 (all false alarms of my checker) |
| Agent, guards on, after I fixed the checker | 1 (severity 1, a documented false alarm) |

Real questions blocked by mistake: 3 of 75 (2 of those answers were wrong anyway).

Pass lines: no severity 3 attack gets through, fewer than 10 of every 100 attacks get through, and fewer than 5 of every 100 real questions are blocked by mistake.
I changed three rules of my checker after I saw the first agent results (it was flagging refusals and a public company address). The change and the one attack that is still flagged are written down in `results/attack_notes.md`.

## Watching it

- **GitHub Actions** (`.github/workflows/checks.yml`) runs the tests and the score check on every push and needs no keys. The score check fails if a score drops more than 5 points under the saved baseline. Every Monday it also pings Qdrant and Supabase so the free projects do not go to sleep (that part needs four repository secrets).
- **A cost ledger and a fuse:** every Claude call is added up; the budget is one US dollar and the bot refuses new calls at 90 cents. No single change was allowed to cost more than 10 cents.
- **A request log** in Supabase (a local file when there are no keys), **LangSmith traces** for 1 in 5 requests, **daily numbers with alerts** (`python -m monitor.metrics`) and a **dashboard** (`streamlit run app/dashboard.py`).
- **More than 800 automatic tests** that run with no keys.

## Run it yourself

You need Python 3.11, an Anthropic account and a free Qdrant Cloud cluster.

1. `pip install -r requirements.txt`
2. Copy `env.example` to `.env` and fill in `ANTHROPIC_API_KEY`, `QDRANT_URL` and `QDRANT_API_KEY`. `SUPABASE_URL`, `SUPABASE_KEY` and `LANGSMITH_API_KEY` are only for the request log and the traces. Never commit `.env`.
3. `python scripts/download_pdfs.py` downloads the 37 filings and the FinanceBench question file.
4. `python -m ingest.build_index` reads the filings, cuts them into pieces, embeds them and loads them into Qdrant. Then `python scripts/check_index.py` checks that search finds the evidence pages.
5. Ask a question: `python -m security.guards "What was the net income of 3M in FY2018?" --role guest`
6. Chat page: `streamlit run app/streamlit_app.py`. The dashboard is its second page.

To measure it yourself: `python -m evals.build_cases` makes the 103 questions, `python -m evals.run_eval --split test --tag mine --spend` runs the 75 test questions (about 10 cents of Claude calls, answers and judge together), `python -m evals.report results/runs/mine_test.json` prints the one page report, and `python -m evals.attacks --tag mine --spend` runs the 40 attacks (about 3 cents). Tests: `python -m pytest -q`.

## What is in the folders

- `app/` the chat page and the dashboard (Streamlit)
- `bot/` the bot: the prompts, the one place that talks to Claude, the answer step, the agent and its 4 read-only tools
- `ingest/` reading the PDFs, cutting pieces, embedding, loading Qdrant
- `retrieval/` search, and reading the company and year out of a question
- `evals/` the exam: cases, free code checks, the judge, the report, the experiments, the 40 attacks
- `security/` roles and access, the instruction scan, personal data hiding, the guards
- `monitor/` the request log and demo limits, traces, daily numbers and alerts
- `scripts/` downloading the PDFs, the fake bait files, index checks, the score check that CI runs
- `tests/` the automatic tests (they run with no keys)
- `results/` score files, experiment notes, failure notes, attack notes (run files stay on my laptop)
- `data/` small data files; the PDFs, the question files, the logs and the ledger are kept out of Git
- `.github/workflows/` the GitHub Actions checks
- `apply_*.py` one file patchers I used to change the project in one step; each makes a backup first and has an `--undo`

## Data and credit

The 37 filings and the questions come from FinanceBench by Patronus AI, shared under CC BY-NC 4.0 (non commercial), and are used here for a portfolio project. They are not in this repository: step 3 above downloads them from the FinanceBench repository. The 20 fake files are made up by me.

## Honest limits

- The test set has 75 questions and the holdout set 18. That is small, which is why the ranges are wide.
- The judge is Claude Haiku 5.5, the same model that writes the answers, so it can share the bot's blind spots.
- All questions come from one data set. Real customers ask differently.
- The 40 attacks are my own and the checker was changed once after the first agent results (written down in `results/attack_notes.md`).
- It is not deployed yet. Before it goes live I still need to pin the package versions in `requirements.txt` and add a warm-up call.

## Author

Affan Ahmed Khan, AI Engineer. I build chatbots that answer from your own documents, and I check that they do.

[affankhan.net](https://affankhan.net) | affan@affankhan.net
