import os
import json
import html

import numpy as np
import pandas as pd
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(
    page_title="SupportSphere AI | Customer Support",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# RGB/neon styling, animated accents, and extra breathing room above the hero.
st.markdown(
r'''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --rgb-cyan: #54ffe3;
  --rgb-blue: #58aaff;
  --rgb-purple: #c27bff;
  --rgb-pink: #ff65c8;
}
html, body, [class*="css"] { font-family: Inter, sans-serif; color: #f6f8ff; }
.stApp {
  background:
    radial-gradient(circle at 5% 8%, rgba(0,255,190,.17), transparent 28%),
    radial-gradient(circle at 95% 9%, rgba(155,70,255,.22), transparent 30%),
    radial-gradient(circle at 52% 95%, rgba(0,155,255,.15), transparent 32%), #080b18;
  color: #f6f8ff;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#101a35 0%,#17102e 55%,#08212b 100%);
  border-right: 1px solid rgba(100,255,225,.3);
}
[data-testid="stSidebar"] * { color: #f2f5ff !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #cbd5f5 !important; }
.block-container { padding-top: 3.1rem; padding-bottom: 2.4rem; max-width: 1500px; }
.hero {
  position: relative; overflow: hidden;
  padding: 27px 30px; border-radius: 25px;
  margin: 14px 0 22px;
  background: linear-gradient(115deg,rgba(0,255,190,.19),rgba(0,157,255,.17),rgba(172,72,255,.24));
  border: 1px solid rgba(143,231,255,.42);
  box-shadow: 0 0 24px rgba(0,225,255,.08), 0 12px 38px rgba(0,0,0,.24);
}
.hero:after {
  content: ""; position: absolute; inset: 0; pointer-events: none; border-radius: inherit;
  background: linear-gradient(110deg,transparent 20%,rgba(255,255,255,.055) 48%,transparent 70%);
  animation: rgbSweep 8s linear infinite;
}
@keyframes rgbSweep { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
.hero-title {
  font-size: clamp(30px,3.5vw,44px); font-weight: 800; line-height: 1.12;
  background: linear-gradient(90deg,#5fffe0,#69baff,#d7a2ff,#ff8bd9);
  background-size: 220% auto; animation: titleGlow 7s ease-in-out infinite alternate;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
@keyframes titleGlow { from { background-position: 0% center; } to { background-position: 100% center; } }
.hero-sub { color: #f0f3ff; font-size: 15px; margin-top: 10px; }
.pill {
  display: inline-block; padding: 7px 12px; border-radius: 999px;
  border: 1px solid rgba(113,255,225,.48); color: #9bffe9;
  background: rgba(0,255,190,.09); font-size: 11px; letter-spacing: .45px;
  font-weight: 800; margin-bottom: 13px;
  box-shadow: 0 0 15px rgba(0,255,190,.08);
}
.metric {
  background: linear-gradient(135deg,rgba(20,28,54,.96),rgba(22,20,52,.94));
  border: 1px solid rgba(145,171,255,.3); padding: 17px 18px;
  border-radius: 17px; min-height: 91px;
  box-shadow: 0 0 17px rgba(87,122,255,.05); transition: transform .2s,border-color .2s,box-shadow .2s;
}
.metric:hover { transform: translateY(-2px); border-color: rgba(84,255,227,.65); box-shadow: 0 0 19px rgba(84,255,227,.09); }
.metric-label { color: #d2dcfa; font-size: 12px; margin-bottom: 7px; }
.metric-value { color: #fff; font-size: 23px; font-weight: 800; }
.stMarkdown, .stMarkdown p, .stMarkdown li, label, [data-testid="stCaptionContainer"] { color: #edf1ff; }
.stMarkdown strong { color: #fff; }
[data-testid="stChatMessage"] {
  background: rgba(18,24,48,.92); border: 1px solid rgba(125,164,255,.28);
  border-radius: 16px; padding: 12px 14px; box-shadow: 0 0 18px rgba(80,130,255,.04);
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: #f5f7ff !important; line-height: 1.7; overflow-wrap: anywhere; }
[data-testid="stExpander"] { background: rgba(15,23,47,.98); border: 1px solid rgba(100,255,218,.3); border-radius: 14px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: #fff !important; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"], [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p, [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li { color: #f4f6ff !important; line-height: 1.75; overflow-wrap: anywhere; }
.stTextInput input, .stTextArea textarea {
  background: rgba(17,24,49,.98) !important; color: #fff !important;
  border: 1px solid rgba(110,173,255,.62) !important; border-radius: 13px !important;
  min-height: 47px; box-shadow: 0 0 0 1px rgba(194,123,255,.08), 0 0 17px rgba(67,165,255,.08);
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--rgb-cyan) !important; box-shadow: 0 0 0 1px var(--rgb-cyan), 0 0 20px rgba(84,255,227,.13) !important; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  border-radius: 12px; border: 1px solid rgba(111,255,226,.58);
  background: linear-gradient(100deg,#00bfa6,#1689e8,#8655e9,#d14cbb);
  background-size: 220% auto; color: #fff; font-weight: 700; min-height: 44px;
  transition: filter .2s, box-shadow .2s, transform .2s;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
  border-color: #b2fff0; color: #fff; filter: brightness(1.13);
  box-shadow: 0 0 19px rgba(91,170,255,.22); transform: translateY(-1px);
}
.chat-panel {
  border: 1px solid rgba(118,151,255,.27); border-radius: 19px;
  padding: 17px 18px 8px; margin-top: 12px;
  background: linear-gradient(135deg,rgba(15,23,47,.48),rgba(26,15,48,.35));
  box-shadow: inset 0 0 22px rgba(93,88,255,.035);
}

.hero-inner { display:flex; align-items:center; gap:24px; position:relative; z-index:1; }
.robot-icon { flex:0 0 112px; width:112px; height:112px; display:flex; align-items:center; justify-content:center;
  font-size:70px; border-radius:28px; background:linear-gradient(145deg,rgba(0,255,224,.12),rgba(139,70,255,.15));
  filter:drop-shadow(0 0 13px rgba(0,225,255,.55)); animation:robotFloat 4s ease-in-out infinite; }
@keyframes robotFloat { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-5px); } }
.hero-copy { min-width:0; }
.hero:before { content:""; position:absolute; width:45%; height:150%; right:-4%; top:-22%; pointer-events:none;
  background:repeating-linear-gradient(155deg,transparent 0 20px,rgba(0,200,255,.09) 21px 23px,transparent 24px 38px);
  mask-image:linear-gradient(90deg,transparent,#000); }
.metric { position:relative; overflow:hidden; display:flex; align-items:center; gap:17px; min-height:116px;
  background:linear-gradient(135deg,rgba(4,32,65,.94),rgba(18,20,58,.96)); }
.metric:after { content:""; position:absolute; inset:0; border-radius:inherit; pointer-events:none;
  background:linear-gradient(110deg,transparent,rgba(255,255,255,.045),transparent); }
.metric:nth-child(1) { border-color:rgba(0,255,220,.65); box-shadow:0 0 15px rgba(0,255,220,.10),inset 0 0 20px rgba(0,255,220,.035); }
.metric-icon { flex:0 0 62px; width:62px; height:62px; display:flex; align-items:center; justify-content:center;
  border-radius:50%; font-size:31px; background:rgba(0,220,255,.12); border:1px solid rgba(0,255,225,.5); box-shadow:0 0 18px rgba(0,220,255,.18); }
.metric-content { position:relative; z-index:1; }
.metric-value { font-size:25px; }
.metric-label { font-size:13px; color:#a9dfff; }
.metric-spark { position:absolute; right:13px; bottom:14px; width:95px; height:28px; opacity:.9; filter:drop-shadow(0 0 5px currentColor); }
.assistant-panel { position:relative; overflow:hidden; margin-top:22px; border-radius:22px; padding:20px 22px 0;
  background:linear-gradient(135deg,rgba(3,25,55,.96),rgba(13,16,54,.97) 60%,rgba(37,12,61,.94));
  border:1px solid rgba(78,108,255,.65); box-shadow:0 0 22px rgba(0,150,255,.12),inset 0 0 26px rgba(0,190,255,.035); }
.assistant-panel:before { content:""; position:absolute; inset:0; padding:1px; border-radius:inherit; pointer-events:none;
  background:linear-gradient(120deg,#00ffe0,#168bff,#713bff,#f02cff); -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0); -webkit-mask-composite:xor; mask-composite:exclude; opacity:.8; }
.assistant-heading { position:relative; z-index:1; font-size:27px; font-weight:800; margin:0 0 7px; color:#54ffe3; }
.assistant-heading span { background:linear-gradient(90deg,#43ffe0,#47baff,#d36cff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.suggestion-line { position:relative; z-index:1; color:#b9d5ff; font-size:14px; margin-bottom:18px; }
.suggestion-line b { color:#51ffe3; }
.feature-strip { display:flex; justify-content:center; align-items:center; gap:0; margin:18px -22px 0; padding:17px 10px;
  background:rgba(1,13,37,.72); border-top:1px solid rgba(85,133,255,.18); position:relative; z-index:1; }
.feature { display:flex; align-items:center; gap:11px; padding:0 28px; min-width:0; }
.feature-icon { font-size:28px; filter:drop-shadow(0 0 7px rgba(0,220,255,.6)); }
.feature-title { font-size:13px; font-weight:800; color:#eef5ff; }
.feature-sub { font-size:11px; color:#a9c5ef; margin-top:3px; }
.feature + .feature { border-left:1px solid rgba(91,139,255,.35); }
[data-testid="stForm"] { position:relative; padding:14px 15px 12px; border-radius:18px; margin-top:-1px;
  background:linear-gradient(135deg,rgba(4,27,59,.96),rgba(12,17,53,.98));
  border:1px solid transparent; background-clip:padding-box;
  box-shadow:0 0 0 1px rgba(0,255,224,.65),0 0 16px rgba(0,180,255,.14),0 0 23px rgba(207,44,255,.10); }
[data-testid="stForm"] [data-testid="stTextInput"] { margin-bottom:8px; }
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button { min-height:48px; font-size:15px; letter-spacing:.15px; }
.stTextInput label { color:#a8d7ff !important; font-weight:700 !important; }
.stTextInput input { font-size:16px !important; }
.stTextInput input::placeholder { color:#8fa9d1 !important; opacity:1 !important; }
.small-muted { color: #cbd5f5; font-size: 12px; }
hr { border-color: rgba(139,165,255,.2); }
footer { visibility: hidden; }
@media (max-width: 700px) {
  .block-container { padding-top: 1.7rem; }
  .hero { padding: 22px 19px; }
  .metric { margin-bottom: 4px; }
  .hero-inner { align-items:flex-start; gap:13px; }
  .robot-icon { flex-basis:64px; width:64px; height:64px; font-size:43px; border-radius:18px; }
  .feature-strip { flex-direction:column; align-items:flex-start; gap:14px; padding-left:22px; }
  .feature { padding:0; }
  .feature + .feature { border-left:0; }
}
</style>''', unsafe_allow_html=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(APP_DIR, "faiss_index.bin")
KB_PATH = os.path.join(APP_DIR, "knowledge_base.csv")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")


@st.cache_data(show_spinner=False)
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "generation_model": "google/flan-t5-small",
        "retrieval_top_k": 4,
    }


@st.cache_data(show_spinner="Loading company knowledge base...")
def load_kb(path):
    data = pd.read_csv(path).fillna("")
    for column in ["instruction", "response"]:
        if column not in data.columns:
            raise ValueError(f"Missing required CSV column: {column}")
    for column in ["category", "intent"]:
        if column not in data.columns:
            data[column] = "General"
    if "document" not in data.columns or not data["document"].astype(str).str.strip().any():
        data["document"] = data.apply(
            lambda row: (
                f"Category: {row['category']}\nIntent: {row['intent']}\n"
                f"Customer question: {row['instruction']}\nReference answer: {row['response']}"
            ),
            axis=1,
        )
    return data


@st.cache_resource(show_spinner="Loading semantic search model...")
def load_embedder(name):
    return SentenceTransformer(name)


@st.cache_resource(show_spinner="Loading answer-generation model...")
def load_generator(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def retrieve(query, top_k, embedder, index, kb):
    if index.ntotal == 0:
        return []
    vector = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    scores, ids = index.search(vector, min(int(top_k), index.ntotal))
    results = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0 or row_id >= len(kb):
            continue
        row = kb.iloc[int(row_id)]
        results.append({
            "score": float(score),
            "question": str(row["instruction"]),
            "answer": str(row["response"]),
            "category": str(row["category"]),
            "intent": str(row["intent"]),
            "document": str(row["document"]),
        })
    return results


def answer_query(query, results, tokenizer, model, device):
    if not results:
        return "I couldn't find relevant information in the available knowledge base. Please contact the support team."
    references = "\n\n".join(
        f"Reference {i}:\n{item['document'][:700]}"
        for i, item in enumerate(results[:4], 1)
    )
    prompt = (
        "You are a polite customer support assistant. Answer only using the reference information. "
        "Do not invent company policies, dates, prices, or guarantees. If references do not contain "
        "enough information, say you cannot confirm it from the available knowledge base. "
        "Keep the answer clear and concise.\n\n"
        f"REFERENCE INFORMATION:\n{references}\n\nCUSTOMER QUESTION:\n{query}\n\nANSWER:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=140, num_beams=2,
            do_sample=False, repetition_penalty=1.1,
        )
    answer = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    return answer or "I couldn't generate a reliable answer from the available references."


def show_sources(sources):
    with st.expander("📚 Retrieved references — source details", expanded=False):
        for number, source in enumerate(sources, 1):
            category = html.escape(str(source.get("category", "General")))
            st.markdown(f"**Source {number} · {category}**")
            st.caption(f"Similarity score: {source['similarity']:.3f}")
            st.markdown(f"**Related question:** {source['question']}")
            st.markdown(f"**Reference answer:**\n\n{source['reference_answer']}")
            if number < len(sources):
                st.divider()


config = load_config()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🌈 SupportSphere AI")
    st.caption("Company Knowledge Assistant")
    st.markdown("---")
    st.markdown("### ⚙️ Assistant settings")
    top_k = st.slider("Retrieved sources", 1, 6, min(6, max(1, int(config.get("retrieval_top_k", 4)))))
    show_refs = st.toggle("Show reference sources", value=True)
    st.markdown("---")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Powered by**")
    st.caption("Hugging Face · Sentence Transformers · FAISS · FLAN-T5")
    st.caption("Prototype using a public customer-support dataset.")

# Neon hero banner with a customer-support robot icon.
st.markdown(
    '''<div class="hero">
      <div class="hero-inner">
        <div class="robot-icon" aria-hidden="true">🤖</div>
        <div class="hero-copy">
          <div class="pill">● RETRIEVAL-AUGMENTED GENERATION · RGB EDITION</div>
          <div class="hero-title">Customer Support AI ✦</div>
          <div class="hero-sub">Ask questions. Retrieve relevant knowledge. Get source-grounded support answers.</div>
        </div>
      </div>
    </div>''',
    unsafe_allow_html=True,
)

if not all(os.path.exists(path) for path in [INDEX_PATH, KB_PATH]):
    st.error("Required files missing. Keep faiss_index.bin and knowledge_base.csv beside app1_corrected.py (or your app script).")
    st.stop()

try:
    kb = load_kb(KB_PATH)
    index = faiss.read_index(INDEX_PATH)
    if index.ntotal != len(kb):
        st.warning(
            f"Index has {index.ntotal:,} vectors but CSV has {len(kb):,} rows. "
            "Use matching artifact files from the same ZIP."
        )
    embedder = load_embedder(config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
except Exception as error:
    st.error(f"Could not load knowledge base or FAISS index: {error}")
    st.stop()

metric_cols = st.columns(3, gap="medium")
metric_data = [
    ("🗄️", "Knowledge records", f"{len(kb):,}", "#00ffe0"),
    ("🕸️", "Indexed vectors", f"{index.ntotal:,}", "#168bff"),
    ("🧠", "Answer model", "FLAN-T5 Small", "#df55ff"),
]
for metric_col, (icon, label, value, accent) in zip(metric_cols, metric_data):
    with metric_col:
        st.markdown(
            f'''<div class="metric" style="border-color:{accent}88;box-shadow:0 0 18px {accent}18,inset 0 0 20px {accent}08;">
              <div class="metric-icon" style="border-color:{accent}88;background:{accent}18;">{icon}</div>
              <div class="metric-content"><div class="metric-label">{label}</div>
              <div class="metric-value">{html.escape(value)}</div></div>
              <svg class="metric-spark" viewBox="0 0 100 28" fill="none" style="color:{accent}"><path d="M2 24 C16 24 15 17 25 21 S39 26 48 13 S62 8 70 17 S85 14 98 3" stroke="{accent}" stroke-width="2.5" stroke-linecap="round"/></svg>
            </div>''',
            unsafe_allow_html=True,
        )

st.markdown(
    '''<div class="assistant-panel">
      <div class="assistant-heading">☏ &nbsp;<span>Ask the knowledge assistant</span></div>
      <div class="suggestion-line">💡 <b>Try:</b> How can I cancel my order? · How can I track my order? · What should I do if my payment failed?</div>
    </div>''',
    unsafe_allow_html=True,
)

# Show previous turns above the input so the user's question and answer stay visible.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and show_refs and message.get("sources"):
            show_sources(message["sources"])

with st.container():
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    with st.form("question_form", clear_on_submit=True, border=False):
        question = st.text_input(
            "Your question",
            placeholder="Type your customer support question here...",
            key="question_input",
            help="Your typed question stays visible in this field until you submit it.",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("➤  Ask SupportSphere", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '''<div class="feature-strip">
          <div class="feature"><div class="feature-icon">🛡️</div><div><div class="feature-title">Source Grounded</div><div class="feature-sub">Accurate &amp; Reliable</div></div></div>
          <div class="feature"><div class="feature-icon">⚡</div><div><div class="feature-title">AI Powered</div><div class="feature-sub">Fast &amp; Smart</div></div></div>
          <div class="feature"><div class="feature-icon">🗄️</div><div><div class="feature-title">RAG Technology</div><div class="feature-sub">Real Knowledge Base</div></div></div>
        </div>''',
        unsafe_allow_html=True,
    )

if submitted:
    prompt = question.strip()
    if not prompt:
        st.warning("Please type a question before submitting.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            with st.spinner("🔎 Searching knowledge and preparing your answer..."):
                results = retrieve(prompt, top_k, embedder, index, kb)
                tokenizer, gen_model, device = load_generator(
                    config.get("generation_model", "google/flan-t5-small")
                )
                answer = answer_query(prompt, results, tokenizer, gen_model, device)
                sources = [
                    {
                        "category": item["category"],
                        "question": item["question"],
                        "reference_answer": item["answer"],
                        "similarity": item["score"],
                    }
                    for item in results
                ]
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as error:
            error_message = f"Sorry, I couldn't process that question. Error: {error}"
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "sources": []}
            )
        # Rerun displays the submitted question and response in the persistent chat history.
        st.rerun()

st.markdown("---")
st.markdown(
    '<div class="small-muted">Prototype based on a public dataset. Answers may not reflect any specific company’s current policies.</div>',
    unsafe_allow_html=True,
)
