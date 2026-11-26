# Failure notes: holdout set

One line for every wrong answer of the holdout run (tag holdout_final, 18 questions, 6 wrong). Only ids, types and counts are written here. The question text and the reference answers belong to FinanceBench (Patronus AI, CC BY-NC 4.0) and stay out of the repository.

How to read a line: the id, the type in brackets, the kind of document, then what happened.
- search missed: the right page was not among the pieces the bot was given.
- misread: the right page was there and the answer is still wrong.
The judge is Claude Haiku 5.5, the same model as the bot, so a mark can be the judge's mistake. Lines that say read it by hand are the ones to check first.

The holdout set was opened once, at the end. Use this file to describe the result, not to tune the bot.
It is a separate file on purpose: the report counts every line of results/failure_notes.md, so the two sets must not be mixed.

Written by apply_final.py from the saved run (26 Nov 2026).

## search missed (4)
- 00283 [search missed] 10-Q: no evidence page among the 4 pieces; the bot answered from other pages; the reference number was not in the answer.
- 00651 [search missed] earnings release: no evidence page among the 5 pieces; the bot answered from other pages; 0 of 2 reference numbers were in the answer.
- 00669 [search missed] 10-K: no evidence page among the 5 pieces; the bot answered from other pages; the reference number was not in the answer.
- 01328 [search missed] 10-K: no evidence page among the 5 pieces; the bot answered from other pages.

## misread: wrong number (1)
- 00799 [misread: wrong number] 10-K: the right page was the 1st of the 4 pieces; 0 of 3 reference numbers were in the answer.

## misread: said not found (1)
- 00540 [misread: said not found] 10-K: the right page was the 1st of the 4 pieces, and the bot still said it could not find the answer.
