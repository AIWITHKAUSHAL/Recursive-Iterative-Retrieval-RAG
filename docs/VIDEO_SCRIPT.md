# YouTube walkthrough (5–7 minutes)

1. Introduce the problem: one-shot retrieval searches once and answers from whatever it finds. Similar wording does not guarantee complete evidence. Multi-part questions often need separate passages; reranking cannot recover documents absent from its candidate pool.
2. Show `data/handbook.json`: H001 explains international work; H002 separately explains probation. Explain that this is a fictional classroom dataset.
3. Show `rag.py`: Euri embeds passages and queries into one embedding space. Cosine similarity orders results. `Pipeline.run` accumulates unique evidence and checks it against the original question. `Euri.assess` returns sufficiency, a reason, missing facts and evidence IDs. `rewrite` targets the missing facts.
4. Show the retry bound: `range(max_retries + 1)`, success break and repeated-query break. Explain why this prevents an infinite loop. If the limit is reached, generation must acknowledge missing evidence.
5. Open the app. Enter your key privately; never record the key or `.env`. Use the default Atlas question, one passage and zero retries. Run and show the incomplete first context and missing-evidence judgment.
6. Set three retries and run again. Expand every attempt. Read the actual original and rewritten queries, scores and texts. Show where H001 and H002 enter the context and the final eligibility answer with citations. Model outputs vary: only claim improvement if the actual trace recovers missing evidence.
7. Run `uv run python evaluate.py`. Show first and final evidence recall and the actual improvement boolean. The first recall is at most 0.5 with one result; final recall reaches 1.0 only if both required passages are recovered. Explain that additional context budget also helps and retries add latency and API cost.
8. Show passing tests and the GitHub repository. Explain that tests use a fake provider for repeatability while the demonstrated app uses live Euri.

## Submission checklist

- Configure your own Euri key locally; never commit `.env`.
- Run the app and live evaluation. Verify a real retry improves evidence before recording.
- Push these project files to the assignment repository. Do not overwrite the reference repository unless intended.
- Record the walkthrough, upload it to YouTube (unlisted is suitable if permitted), and submit the GitHub and video links.
- Recording and YouTube publication require your account and are not performed by this code.
