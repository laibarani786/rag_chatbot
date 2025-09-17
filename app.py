"""
RAG Retrieval Streamlit App (Task 3) — Modified
Features:
- Load knowledge_base.txt (fallback default facts)
- Upload custom .txt knowledge base
- sentence-transformers embeddings (all-MiniLM-L6-v2)
- FAISS vector index (IndexFlatIP) with normalized embeddings (cosine via inner product)
- Similarity search returning top-k facts with percentage score
- Sidebar replaced with system prompt input
- Simple text results, no highlights
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# -------------------------------
# Helpers
# -------------------------------

def default_facts():
    return [
        "GlobalMart sells electronics including laptops, smartphones, and tablets.",
        "GlobalMart provides free delivery on orders above $100.",
        "GlobalMart offers a 2-year warranty on all electronics.",
        "GlobalMart has a return policy of 30 days for all products.",
        "GlobalMart also sells groceries, household supplies, and personal care items.",
        "GlobalMart gives a 10% student discount on all purchases.",
        "GlobalMart has physical stores in New York, Los Angeles, and Chicago.",
        "GlobalMart's customer support is available 24/7 via phone and email.",
        "GlobalMart offers seasonal discounts during Black Friday and Cyber Monday.",
    ]


def read_kb_file(file_path="knowledge_base.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            return lines
    except Exception:
        pass
    return default_facts()


# -------------------------------
# Cached resources: model + index
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_index(facts):
    model = load_model()
    embeddings = model.encode(facts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized = embeddings / norms
    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized.astype('float32'))
    return index, normalized


def cosine_scores_from_index(question, model, index, facts, top_k=3):
    q_emb = model.encode([question], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    D, I = index.search(q_emb.astype('float32'), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(facts):
            continue
        pct = float((score + 1) / 2 * 100)
        results.append({"fact": facts[idx], "score": float(score), "pct": pct})
    return results


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="GlobalMart RAG Retrieval", layout="wide")

# Header / Banner
with st.container():
    cols = st.columns([1, 6, 1])
    with cols[0]:
        st.image("https://raw.githubusercontent.com/your-repo/globalmart-logo/main/logo.png", width=80)
    with cols[1]:
        st.markdown("""
        <h1 style='margin:0; padding:0;'>🔎 GlobalMart — RAG Retrieval Demo</h1>
        <p style='margin:0; color:gray;'>Ask questions about GlobalMart and get the most relevant facts from the knowledge base.</p>
        """, unsafe_allow_html=True)
    with cols[2]:
        if st.button("🆕 New Chat", key="new_chat_header"):
            st.session_state.clear()


# Sidebar: system prompt + top-k + about
with st.sidebar:
    st.header("💡 System Prompt")
    user_prompt = st.text_area("Enter your system prompt (this will guide the search)", height=150)
    top_k = st.slider("Top results to return", min_value=1, max_value=5, value=3)
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown(
        "This demo builds the **Retrieval** part of a RAG system using `sentence-transformers` and `FAISS`.\n\n"
        "Use the system prompt above to guide your queries. Click 'New Chat' to reset the app."
    )


# Main area: load KB and model, build index
kb_from_file = None
uploaded = st.file_uploader("Upload a .txt knowledge base (optional)", type=["txt"])
if uploaded is not None:
    try:
        raw = uploaded.read().decode('utf-8')
        kb_from_file = [line.strip() for line in raw.splitlines() if line.strip()]
    except Exception:
        st.error("Could not read uploaded file — make sure it's a UTF-8 .txt file.")

facts = kb_from_file if kb_from_file is not None else read_kb_file()
st.sidebar.success(f"Using {len(facts)} facts")

with st.spinner("Loading model & building index (fast)..."):
    model = load_model()
    try:
        index, embeddings = build_index(facts)
    except Exception as e:
        st.error("Error building FAISS index: " + str(e))
        st.stop()


# Input area
st.subheader("Ask a question about GlobalMart")
col1, col2 = st.columns([4,1])
with col1:
    user_q = st.text_input("Type your question here", key="user_question")
with col2:
    if st.button("Search", key="search_button"):
        st.session_state['_do_search'] = True

# Clear button
if st.button("🧹 Clear", key="clear_main"):
    st.session_state.clear()
    st.rerun()


# Perform search
search_query = user_prompt if user_prompt and user_prompt.strip() else user_q

if st.session_state.get('_do_search') and search_query and search_query.strip():
    with st.spinner("Searching..."):
        results = cosine_scores_from_index(search_query, model, index, facts, top_k=top_k)

    if not results:
        st.warning("No relevant facts found.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        for i, r in enumerate(results, start=1):
            st.markdown(f"**Result {i} ({r['pct']:.1f}% match):** {r['fact']}")
        st.markdown("---")
        st.success("Search complete — results above.")

elif st.session_state.get('_do_search') and (not search_query or not search_query.strip()):
    st.warning("Please enter a question or system prompt before clicking Search.")

# Footer / tips
with st.container():
    st.markdown("---")
    st.markdown(
        "**Tips:** Try questions like: 'Do you offer discounts?', 'What is the return policy?', or 'Where are your stores located?'"
    )
