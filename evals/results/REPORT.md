| | baseline-main | agent-v0.4.0-first-run | agent-main-rules | agent-main-auto |
|---|---|---|---|---|
| Test set | main (21 questions × 1) | main (21 questions × 2) | main (21 questions × 2) | main (21 questions × 2) |
| Accuracy (all) | 11/21 (52%) | 20/42 (48%) | 24/42 (57%) | 14/42 (33%) |
| Single-fact | 4/7 (57%) | 6/14 (43%) | 8/14 (57%) | 4/14 (29%) |
| Multi-step | 3/10 (30%) | 6/20 (30%) | 8/20 (40%) | 2/20 (10%) |
| Unanswerable → 'not found' | 4/4 (100%) | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| Made-up answers | 0 | 0 | 0 | 0 |
| Right page found | 14/18 (78%) | 28/36 (78%) | 34/36 (94%) | 33/36 (92%) |
| Cited but wrong | 4 | 10 | 10 | 5 |
| Median time | 2.8 s | 13.7 s | 15.2 s | 17.2 s |
| Tokens | 17,439 | 132,077 | 213,578 | 257,816 |
| Cost | $0.0000 | $0.0000 | $0.0000 | $0.0000 |
| Model | qwen2.5:7b | qwen2.5:7b | qwen2.5:7b | qwen2.5:7b |
| Errors | 0 | 0 | 0 | 0 |
| Same result every run | – | 21/21 (100%) | 21/21 (100%) | 21/21 (100%) |
| Claim checker | – | rules | rules | auto |

## Per question

| id | type | baseline-main | agent-v0.4.0-first-run | agent-main-rules | agent-main-auto |
|---|---|---|---|---|---|
| s01 | single | ✗ | ✓ ✓ | ✗ ✗ | ✗ ✗ |
| s02 | single | ✓ | ✗ ✗ | ✓ ✓ | ✓ ✓ |
| s03 | single | ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| s04 | single | ✗ | ✓ ✓ | ✓ ✓ | ✗ ✗ |
| s05 | single | ✓ | ✗ ✗ | ✓ ✓ | ✗ ✗ |
| s06 | single | ✓ | ✗ ✗ | ✗ ✗ | ✗ ✗ |
| s07 | single | ✗ | ✗ ✗ | ✗ ✗ | ✗ ✗ |
| m01 | multi | ✗ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| m02 | multi | ✗ | ✗ ✗ | ✓ ✓ | ✗ ✗ |
| m03 | multi | ✗ | ✗ ✗ | ✗ ✗ | ✗ ✗ |
| m04 | multi | ✗ | ✗ ✗ | ✓ ✓ | ✗ ✗ |
| m05 | multi | ✓ | ✗ ✗ | ✗ ✗ | ✗ ✗ |
| m06 | multi | ✓ | ✗ ✗ | ✗ ✗ | ✗ ✗ |
| m07 | multi | ✗ | ✗ ✗ | ✗ ✗ | ✗ ✗ |
| m08 | multi | ✗ | ✓ ✓ | ✓ ✓ | ✗ ✗ |
| m09 | multi | ✗ | ✓ ✓ | ✗ ✗ | ✗ ✗ |
| m10 | multi | ✓ | ✗ ✗ | ✗ ✗ | ✗ ✗ |
| u01 | unanswerable | ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| u02 | unanswerable | ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| u03 | unanswerable | ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| u04 | unanswerable | ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
