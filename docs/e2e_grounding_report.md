# E2E Grounding Evaluation

- Mode: `ollama_generation`
- Result: **9/10 passed** (90.0%)
- Source compliance: **10/10**
- Evidence-term support: **9/10**

The pass rule requires a retrieved source to appear in the answer, every bracketed citation to be retrieved, and at least two expected evidence terms to appear.
This is a reproducible lexical check, not a complete semantic proof that no unsupported claim was generated.

| ID | Pass | Sources | Evidence terms | Failure |
| --- | --- | --- | --- | --- |
| Q01 | yes | docs/response_guidance.md | 3 |  |
| Q02 | yes | docs/response_guidance.md | 3 |  |
| Q03 | no | docs/response_guidance.md | 0 | fewer than two expected evidence terms are present |
| Q04 | yes | docs/response_guidance.md | 3 |  |
| Q05 | yes | docs/response_guidance.md | 3 |  |
| Q06 | yes | docs/response_guidance.md | 2 |  |
| Q07 | yes | docs/response_guidance.md | 3 |  |
| Q08 | yes | docs/response_guidance.md | 3 |  |
| Q09 | yes | docs/response_guidance.md | 2 |  |
| Q10 | yes | docs/response_guidance.md | 2 |  |
