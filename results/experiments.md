# Experiments

Every setting tried, one at a time. Experiments 1 to 5 are scored on search alone.

### Experiment 1: Keyword search (BM25) mixed into vector search (2026-11-23 13:08)
85 questions, search only, filter company+year.

| variant | right page found | MRR | first place | all pages found (of 16) |
|---|---|---|---|---|
| vector only | 51 of 85 | 0.397 | 24 of 85 | 1 of 16 |
| keyword only | 24 of 85 | 0.186 | 12 of 85 | 0 of 16 |
| both mixed | 39 of 85 | 0.291 | 17 of 85 | 0 of 16 |

Chosen: vector search only. Keyword search alone finds the right page for 24 of 85 questions, and mixing it in lowers vector search from 51 to 39.

### Experiment 2: Pieces returned (top k): 3, 4, 5 and 8 (2026-11-23 13:12)
85 questions, search only, filter company+year.

| variant | right page found | MRR | first place | all pages found (of 16) |
|---|---|---|---|---|
| top 3 | 40 of 85 | 0.369 | 24 of 85 | 0 of 16 |
| top 4 | 45 of 85 | 0.383 | 24 of 85 | 0 of 16 |
| top 5 | 51 of 85 | 0.397 | 24 of 85 | 1 of 16 |
| top 8 | 54 of 85 | 0.403 | 24 of 85 | 3 of 16 |

Chosen: the best 5 pieces (TOP_K = 5). The best 3 and the best 4 find fewer pages (40 and 45 of 85). The best 8 find 3 more (54) but send 3 more pieces to Claude for every question.

### Experiment 4: Company filter: off, company, company+year (2026-11-23 13:15)
85 questions, search only, filter company+year.

| variant | right page found | MRR | first place | all pages found (of 16) |
|---|---|---|---|---|
| filter off | 43 of 85 | 0.355 | 23 of 85 | 1 of 16 |
| filter company | 46 of 85 | 0.377 | 24 of 85 | 1 of 16 |
| filter company+year | 51 of 85 | 0.397 | 24 of 85 | 1 of 16 |

Chosen: the filter on company and year (51 of 85). The filter on company alone finds 46 and no filter finds 43.

### Experiment 5: Query rewrite by Claude: off or on (2026-11-23 13:38)
85 questions, search only, filter company+year. The rewrites cost 0.354 cents.

| variant | right page found | MRR | first place | all pages found (of 16) |
|---|---|---|---|---|
| no rewrite | 51 of 85 | 0.397 | 24 of 85 | 1 of 16 |
| rewrite by Claude | 64 of 85 | 0.572 | 39 of 85 | 8 of 16 |

Chosen: the query rewrite on. It lifts the right page from 51 to 64 of 85 and the rank score from 0.397 to 0.572 for 0.354 cents.

### Experiment 8: Query words: rewrite, finance word hints, up to 3 queries (2026-11-25 13:42)
85 questions, search only, filter company+year. The rewrites cost 2.171 cents.

| variant | right page found | MRR | first place | all pages found (of 16) |
|---|---|---|---|---|
| no rewrite | 51 of 85 | 0.397 | 24 of 85 | 1 of 16 |
| rewrite by Claude | 64 of 85 | 0.572 | 39 of 85 | 8 of 16 |
| rewrite + word hints | 66 of 85 | 0.636 | 47 of 85 | 8 of 16 |
| up to 3 queries + word hints | 64 of 85 | 0.555 | 36 of 85 | 7 of 16 |

Chosen: one rewritten query with the finance word hints (rewrite_v2): 66 of 85 and rank score 0.636. Three queries per question find 64 and rank worse (0.555), so QUERY_COUNT stays 1.

### Rewrite test (2026-11-25 21:27)
85 questions, search only, filter company+year.

| variant | right page found | MRR | first place |
|---|---|---|---|
| question as typed (no rewrite) | 51 of 85 | 0.397 | 24 of 85 |
| now: rewrite_v2 with word hints | 66 of 85 | 0.636 | 47 of 85 |
| now, with the noise taken out | 66 of 85 | 0.636 | 47 of 85 |

Chosen: rewrite_v2 as it is. Taking the noise out of the rewritten query changes nothing (66 of 85 both ways), so QUERY_CLEAN stays False.

### Rewrite test (2026-11-26 10:44)
85 questions, search only, filter company+year. The new rewrites cost 1.125 cents.

| variant | right page found | MRR | first place |
|---|---|---|---|
| question as typed (no rewrite) | 51 of 85 | 0.397 | 24 of 85 |
| now: rewrite_v2 with word hints | 66 of 85 | 0.636 | 47 of 85 |
| now, with the noise taken out | 66 of 85 | 0.636 | 47 of 85 |
| new: rewrite_v3 line, noise taken out | 63 of 85 | 0.625 | 47 of 85 |

Chosen: rewrite_v2 stays. rewrite_v3 finds 63 of 85 (rank score 0.625) against 66 (0.636) for v2, and ties on first place (47).

### Experiment 3 and the Colab lab: header line, large model, context lines, piece sizes (2026-11-26 12:00)
85 questions, search only, filter company+year. Scored in Google Colab with the saved rewrites, so it cost 0 Claude cents.
"Bot reads" is the right page among the first 3 different pages of the best 5 pieces, the pages the bot reads whole. It decides.

| collection | what changed | bot reads | best 5 | MRR | first place |
|---|---|---|---|---|---|
| filings_v1 | nothing (today) | 62 of 85 | 66 of 85 | 0.636 | 47 of 85 |
| noheader | header line off | 45 of 85 | 50 of 85 | 0.478 | 35 of 85 |
| large | large model (1024 numbers, not 384) | 61 of 85 | 70 of 85 | 0.602 | 40 of 85 |
| context | context lines added | 55 of 85 | 58 of 85 | 0.522 | 35 of 85 |
| size250 | pieces of 250 tokens | 56 of 85 | 63 of 85 | 0.578 | 41 of 85 |
| size480 | pieces of 480 tokens | 62 of 85 | 65 of 85 | 0.623 | 46 of 85 |

Chosen: filings_v1 stays (header line on, small model, pieces of about 400 tokens). Nothing beat it by 2 questions.
The header line is worth 17 questions (62 against 45). The large model finds the page among the best 5 more often (70 against 66)
but ranks it lower, so the pages the bot reads match fewer (61 against 62). Context lines lost 7, pieces of 250 tokens lost 6, and pieces of 480 tokens tie.
