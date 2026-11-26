# Failure notes: test set

One line for every wrong answer of the test run (tag fixes, 75 questions, 22 wrong). Only ids, types and counts are written here. The question text and the reference answers belong to FinanceBench (Patronus AI, CC BY-NC 4.0) and stay out of the repository.

How to read a line: the id, the type in brackets, the kind of document, then what happened.
- search missed: the right page was not among the pieces the bot was given.
- misread: the right page was there and the answer is still wrong.
The judge is Claude Haiku 5.5, the same model as the bot, so a mark can be the judge's mistake. Lines that say read it by hand are the ones to check first.

The report (python -m evals.report) counts the types in this file.

Written by apply_final.py from the saved run (25 Nov 2026).

## search missed (10)
- 00382 [search missed] earnings release: no evidence page among the 5 pieces; the bot said it could not find the answer; the reference number was not in the answer.
- 00601 [search missed] earnings release: no evidence page among the 5 pieces; the bot answered from other pages; the reference number was not in the answer.
- 00724 [search missed] 10-Q: no evidence page among the 4 pieces; the bot answered from other pages; the reference answer has no number to check.
- 00757 [search missed] 10-K: no evidence page among the 4 pieces; the bot said it could not find the answer; the reference number was not in the answer.
- 01028 [search missed] 10-K: no evidence page among the 5 pieces; the bot answered from other pages; the reference answer has no number to check.
- 01290 [search missed] 10-K: no evidence page among the 5 pieces; the bot answered from other pages; the reference number was not in the answer.
- 01928 [search missed] earnings release: no evidence page among the 5 pieces; the bot said it could not find the answer; the reference number was not in the answer.
- 01981 [search missed] 10-K: no evidence page among the 5 pieces; the bot said it could not find the answer; the reference answer has no number to check.
- 02119 [search missed] 10-Q: no evidence page among the 4 pieces; the bot answered from other pages; the reference number was not in the answer.
- 02419 [search missed] 10-Q: no evidence page among the 4 pieces; the bot answered from other pages; the reference answer has no number to check.

## misread: wrong number (7)
- 00005 [misread: wrong number] 10-K: the right page was the 1st of the 4 pieces; the reference number was not in the answer.
- 00215 [misread: wrong number] 10-K: the right page was the 1st of the 5 pieces; the reference number was not in the answer.
- 00460 [misread: wrong number] 10-Q: the right page was the 2nd of the 5 pieces; 0 of 3 reference numbers were in the answer.
- 00517 [misread: wrong number] 10-K: the right page was the 1st of the 5 pieces; 0 of 3 reference numbers were in the answer; the answer cited another page.
- 00790 [misread: wrong number] 10-K: the right page was the 3rd of the 5 pieces; 0 of 3 reference numbers were in the answer.
- 01107 [misread: wrong number] 10-K: the right page was the 1st of the 4 pieces; 0 of 2 reference numbers were in the answer; the answer cited another page.
- 01902 [misread: wrong number] 10-Q: the right page was the 2nd of the 3 pieces; the reference number was not in the answer; the answer cited another page.

## misread: said not found (2)
- 00606 [misread: said not found] earnings release: the right page was the 1st of the 2 pieces, and the bot still said it could not find the answer.
- 01911 [misread: said not found] earnings release: the right page was the 1st of the 4 pieces, and the bot still said it could not find the answer.

## misread: answer differs (2)
- 00995 [misread: answer differs] 10-K: the right page was the 4th of the 5 pieces; the reference answer has no number to check, so only the judge could mark it.
- 01009 [misread: answer differs] 10-K: the right page was the 3rd of the 4 pieces; the reference answer has no number to check, so only the judge could mark it.

## misread: right number, marked wrong (1)
- 10420 [misread: right number, marked wrong] 10-K: the right page was the 1st of the 4 pieces; every reference number was in the answer, but the judge marked it wrong, so read it by hand.
