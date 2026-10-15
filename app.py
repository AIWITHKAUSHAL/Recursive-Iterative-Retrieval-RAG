"""Streamlit workbench for inspecting every retrieval attempt."""
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import Chunk, Euri, Pipeline, chunk_text

load_dotenv()
ROOT = Path(__file__).parent
DEMO = "Can an Atlas employee who joined 20 days ago work remotely from Lisbon for 20 days, and who must approve it?"
st.set_page_config(page_title="Retrace | Iterative RAG", page_icon="🔎", layout="wide")
st.title("Retrace · Iterative retrieval RAG")
st.caption("Retrieve → check evidence → rewrite → retrieve again → grounded answer")
with st.sidebar:
    st.header("Live Euri connection")
    key = st.text_input("Euri API key", value=os.getenv("EURI_API_KEY", ""), type="password")
    st.caption("Questions and documents are sent to Euri. Keys stay in this server session.")
    retries = st.slider("Maximum retries", 0, 5, 3)
    top_k = st.slider("Passages per attempt", 1, 10, 1)
    st.caption("Total searches ≤ 1 + maximum retries. Use zero retries for one-shot retrieval.")
    uploads = st.file_uploader("Add UTF-8 text / Markdown", type=["txt", "md"], accept_multiple_files=True)
    include_sample = st.checkbox("Include fictional Atlas handbook", value=True)
    st.caption("A relevance judgment is a model estimate, not a guarantee of correctness.")
st.info("Demo: the international work policy and probation eligibility are separate passages. With one passage per attempt, the first search cannot supply both.")
question = st.text_area("Original question", value=DEMO, max_chars=4000)
chunks = [Chunk(**item) for item in json.loads((ROOT / "data/handbook.json").read_text())] if include_sample else []
try:
    for i, upload in enumerate(uploads):
        if upload.size > 1_000_000:
            raise ValueError("Each upload must be at most 1 MB.")
        chunks.extend(chunk_text(upload.getvalue().decode("utf-8"), upload.name, f"U{i}_"))
    if len(chunks) > 200:
        raise ValueError("Use at most 200 chunks per run.")
except (ValueError, UnicodeDecodeError) as exc:
    st.error(str(exc))
    st.stop()
with st.expander(f"Knowledge base · {len(chunks)} chunks"):
    st.json([vars(c) for c in chunks])
if st.button("Run live retrieval", type="primary"):
    st.session_state.pop("result", None)
    try:
        with st.status("Embedding knowledge base and checking evidence…", expanded=True) as status:
            provider = Euri(key)
            pipeline = Pipeline(provider, chunks)
            result = pipeline.run(question, retries, top_k,
                on_attempt=lambda a: st.write(f"Attempt {a['attempt']}: {a['query']} — {a['assessment']['reason']}"))
            result["models"] = {"embedding": provider.embedding_model, "generation": provider.chat_model}
            st.session_state.result = result
            status.update(label="Retrieval complete", state="complete")
    except Exception as exc:
        # Do not show raw provider bodies, request headers, or credentials.
        st.error(str(exc) if isinstance(exc, (ValueError, RuntimeError)) else
                 f"Run failed ({type(exc).__name__}). Check connection and model configuration.")
if "result" in st.session_state:
    r = st.session_state.result
    st.subheader("Final answer")
    st.markdown(r["answer"])
    if not r["sufficient"]:
        st.warning("Evidence remains incomplete. Read the missing-evidence notes below.")
    st.caption(f"Stop: {r['stop_reason']} · {len(r['attempts'])} searches")
    st.write("**Original query:**", r["original_query"])
    for attempt in r["attempts"]:
        with st.expander(f"Attempt {attempt['attempt']} · {'Initial query' if attempt['attempt']==1 else 'Rewritten query'}", expanded=True):
            st.write(attempt["query"])
            st.json(attempt["assessment"])
            for hit in attempt["results"]:
                st.markdown(f"**#{hit['rank']} · [{hit['chunk']['id']}] {hit['chunk']['source']}** · cosine {hit['score']:.4f}")
                st.write(hit["chunk"]["text"])
            st.caption("Accumulated context: " + ", ".join(attempt["context_ids"]))
    with st.expander("Final context sent to generation", expanded=True):
        for hit in r["final_context"]:
            st.write(f"[{hit['chunk']['id']}] {hit['chunk']['source']}")
            st.write(hit["chunk"]["text"])
    st.download_button("Download full trace JSON", json.dumps(r, indent=2), "retrieval-trace.json", "application/json")
