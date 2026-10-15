# Diagram prompts

Final assets: `iterative-rag-concept.png` and `application-flow.png`.

The concept image gives the clearest view of stop conditions; the application image maps those stages to the source files and Euri services.

Generated with the built-in imagegen tool. These images explain the concept and the application architecture, rather than benchmark outcomes.

## Concept image

Use case: scientific-educational / infographic-diagram.
Create a beautiful, exceptionally clear educational infographic image, landscape 16:9, high resolution, suitable for a YouTube explanation and university assignment. Title exactly: "Recursive / Iterative Retrieval RAG". Subtitle: "Search again when the evidence is incomplete".
Polished editorial vector-like illustration, light ivory background, dark navy text, teal for retrieval, amber for missing evidence and retry arrows, green for sufficient evidence. Large crisp typography, generous spacing, simple meaningful icons (question bubble, document stack, magnifying glass, checklist, pencil, answer). Readable at presentation size. No ornamental clutter.
The main diagram occupies upper 70 percent. The main path has these exact labelled stages with unmistakable directional arrows: "Original question" → "Retrieve passages" → "Accumulate unique evidence" → decision diamond "Enough evidence?".
From that decision a green YES arrow reaches "Generate cited answer".
An amber NO arrow descends to decision "Retries left?". YES goes to "Rewrite for missing facts", and a thick amber return loop goes back specifically to "Retrieve passages", labelled "Search again". NO goes to "Partial answer + missing facts". The partial-answer branch must NOT go back into retrieval.
Place a compact safety badge: "Stop at retry limit or repeated query".
Bottom explanatory strip with three separated panels:
1 "Why one-shot can fail" / "One result may cover only part of the question."
2 "What improves" / "Rewrites target gaps; evidence accumulates."
3 "Bounded loop" / "Total searches ≤ 1 + max retries"
Add a tiny concrete example beneath the first two panels if space allows: "International work policy + probation policy → complete eligibility evidence".
Semantically exact: always judge accumulated context against original question; looping is conditional; improvement is not guaranteed. Do not invent model names, metrics, or recursive function calls. The phrase recursive / iterative is broad: depict actual iterative search, not recursion tree. All arrows must have clear arrowheads, do not cross text. Text must be spelled accurately.

## Application image

Use case: infographic-diagram.
Create a professional, beautifully laid out high resolution landscape 16:9 educational architecture infographic, matching an ivory/navy/teal/amber/green editorial visual language. Main title exactly "Retrace: Application Flow". Subtitle "From documents to a transparent, grounded answer". Use large crisp text, meaningful simple icons, ample whitespace and accurately routed arrows. For a real Streamlit Python application using live Euri API calls. No fake web interface screenshot.
Organize into THREE clearly labelled horizontal lanes:
TOP lane "1 · Build the knowledge base": "Atlas sample / TXT / MD uploads" → "Validate and chunk text" → "Euri document embeddings" → "In-memory vector index". Small label under chunking "Stable chunk IDs"; under index "Normalized vectors".
MIDDLE lane "2 · Run iterative retrieval": an input card "Question + top k + max retries" → "Embed search query" → "Cosine top-k search" → "Merge unique passages" → diamond "Evidence sufficient?".
The vector index feeds the cosine search through one clear short connector.
YES from the sufficient diamond routes toward the generation stage in bottom lane.
NO goes through a visible amber mini-branch "Retries remain + query is new?" then to "Euri query rewrite" and a prominent amber return loop to "Embed search query". If no retries or repeated query, the amber exit routes to generation with label "Use available evidence; state gaps". Preserve semantic truth: repeated-query check is effectively part of loop guard; never imply a new infinite cycle.
BOTTOM lane "3 · Answer and inspect": "Euri answer generation" → "Validate citation IDs" → "Streamlit results + JSON trace".
Under results a concise list: "Original and rewritten queries • Every ranked result • Evidence checks • Final context • Stop reason".
Add tasteful code-file badges: "app.py" next to inputs and results; "rag.py · Pipeline.run" beside loop; "rag.py · Euri" beside model services.
Footer info chips: "Live embeddings + judging + rewriting + generation via Euri" and "At most 1 + max retries searches".
Make clear every sufficiency judgment uses original question plus accumulated passages. Include a small label near sufficient decision "Original question + accumulated context". Do not depict FAISS, databases, LangChain, reranking, agents, tools, PDF uploads, or streaming responses; these are absent from this implementation. No extra architecture components. Short labels and correctly directed, non-overlapping arrows are more important than decoration.
