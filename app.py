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
st.set_page_config(page_title="Retrace | Retrieval RAG", page_icon="🔎", layout="wide")
st.title("Retrace · Iterative and recursive retrieval RAG")
st.caption("Iterative query refinement or recursive subquestion retrieval → grounded answer")
with st.sidebar:
    st.header("Live Euri connection")
    key = st.text_input("Euri API key", value=os.getenv("EURI_API_KEY", ""), type="password")
    st.caption("Questions and documents are sent to Euri. Keys stay in this server session.")
    mode = st.segmented_control("Retrieval approach", ["Iterative", "Recursive"], default="Iterative")
    if mode == "Recursive":
        max_depth = st.slider("Maximum recursion depth", 0, 4, 2)
        max_searches = st.slider("Maximum total searches", 1, 20, 6)
        st.caption("The root is depth 0. Subquestions share evidence and the total search budget.")
    else:
        retries = st.slider("Maximum retries", 0, 5, 3)
        st.caption("Total searches ≤ 1 + maximum retries. Use zero retries for one-shot retrieval.")
    top_k = st.slider("Passages per attempt", 1, 10, 1)
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
            def show_attempt(a):
                st.write(f"Attempt {a['attempt']}: {a['query']} — {a['assessment']['reason']}")
            if mode == "Recursive":
                result = pipeline.run_recursive(question, max_depth, max_searches, top_k,
                                                on_attempt=show_attempt)
            else:
                result = pipeline.run(question, retries, top_k, on_attempt=show_attempt)
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
        assessment = r.get("final_assessment", r["attempts"][-1]["assessment"])
        st.warning("The retrieved passages do not yet support a complete answer.")
        st.write("**Missing evidence:**", assessment.get("missing") or
                 "The evidence judge did not specify the missing facts.")
        st.write("**Evidence check:**", assessment["reason"])
        stop_notes = {
            "retry_limit": "The iterative retry budget was exhausted. Increase Maximum retries or Passages per attempt, then run again.",
            "search_limit": "The recursive search budget was exhausted. Increase Maximum total searches or Passages per attempt, then run again.",
            "depth_limit": "A recursive branch reached the depth limit. Increase Maximum recursion depth or Passages per attempt, then run again.",
            "repeated_query": "The model repeated an earlier search. Try a more specific original question or increase Passages per attempt.",
            "decomposition_exhausted": "The subquestion tree finished without enough evidence. Try a more specific question or increase Passages per attempt.",
        }
        st.info(stop_notes.get(r["stop_reason"], "Inspect the final evidence check and retrieved passages below."))
        st.caption("If the missing facts are absent from the knowledge base, upload a document containing them. More searches cannot supply facts absent from the documents.")
    st.caption(f"Mode: {r.get('mode', 'iterative')} · Stop: {r['stop_reason']} · {len(r['attempts'])} searches")
    st.write("**Original query:**", r["original_query"])
    if r.get("mode") == "recursive":
        with st.expander("Recursive subquestion tree", expanded=True):
            for node in r["nodes"]:
                st.text(f"{'  ' * node['depth']}Node {node['id']} (parent {node['parent_id']}, "
                        f"depth {node['depth']}): {node['query']} [{node['status']}]")
    for attempt in r["attempts"]:
        label = (f"Subquestion at depth {attempt['depth']}" if 'depth' in attempt
                 else 'Initial query' if attempt['attempt'] == 1 else 'Rewritten query')
        with st.expander(f"Attempt {attempt['attempt']} · {label}", expanded=True):
            st.write(attempt["query"])
            st.json(attempt["assessment"])
            if "local_assessment" in attempt:
                st.write("Subquestion evidence check:")
                st.json(attempt["local_assessment"])
            for hit in attempt["results"]:
                st.markdown(f"**#{hit['rank']} · [{hit['chunk']['id']}] {hit['chunk']['source']}** · cosine {hit['score']:.4f}")
                st.write(hit["chunk"]["text"])
            st.caption("Accumulated context: " + ", ".join(attempt["context_ids"]))
    with st.expander("Final context sent to generation", expanded=True):
        for hit in r["final_context"]:
            st.write(f"[{hit['chunk']['id']}] {hit['chunk']['source']}")
            st.write(hit["chunk"]["text"])
    st.download_button("Download full trace JSON", json.dumps(r, indent=2), "retrieval-trace.json", "application/json")
