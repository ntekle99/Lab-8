import os
import re
import string
from collections import Counter

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from htmlTemplates import css, bot_template, user_template


# ----------------------------
# Helpers: PDF -> text
# ----------------------------
def read_pdf_files(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        except Exception as e:
            st.error(f"Failed to read {pdf.name}: {e}")
    return text


# ----------------------------
# Helpers: chunking for resumes (smaller chunks for short docs)
# ----------------------------
def chunk_text(text, chunk_size=300, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)


# ----------------------------
# Helpers: Vector store
# ----------------------------
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY or base from env
    store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return store


# ----------------------------
# Reviewer Prompt
# ----------------------------
REVIEW_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert, constructive, and precise resume reviewer. "
        "You ONLY use the provided context (from the candidate's resume and any attached notes) "
        "to answer. When needed, propose specific rewrites with measurable outcomes, strong verbs, "
        "and targeted keywords for the intended roles. If something is missing from the context, "
        "say so and suggest what to add.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n"
    ),
)


# ----------------------------
# LLM + Conversational RAG
# ----------------------------
def make_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)  # deterministic feedback
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": REVIEW_PROMPT},
    )
    return chain


# ----------------------------
# Lightweight keyword extraction (no heavy deps)
# ----------------------------
_STOPWORDS = set("""
a an and are as at be by for from has have in into is it its of on or that the their to was were will with you your
""".split())

def normalize_tokens(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t and t not in _STOPWORDS and not t.isdigit()]
    return tokens

def top_keywords(text, top_k=30):
    tokens = normalize_tokens(text)
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_k)]

def highlight_gaps(resume_text, jd_text, top_k=30):
    """Return (resume_top, jd_top, missing_from_resume) keyword sets."""
    resume_kw = set(top_keywords(resume_text, top_k=top_k))
    jd_kw = set(top_keywords(jd_text, top_k=top_k))
    missing = [w for w in sorted(jd_kw - resume_kw)]
    return sorted(resume_kw), sorted(jd_kw), missing


# ----------------------------
# Chat UI rendering
# ----------------------------
def render_chat(chat_history):
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# ----------------------------
# Streamlit App
# ----------------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="AI Resume Reviewer", page_icon="💼", layout="wide")
    st.write(css, unsafe_allow_html=True)

    st.title("💼 AI Resume Reviewer")
    st.caption("Upload your resume (and optionally a job description) then chat for targeted feedback.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""

    # ---------------- Sidebar: Upload & Process ----------------
    with st.sidebar:
        st.header("📄 Upload")
        resume_pdfs = st.file_uploader(
            "Upload your resume PDF(s)",
            accept_multiple_files=True,
            type=["pdf"]
        )
        jd_pdfs = st.file_uploader(
            "Optional: Upload Job Description PDF(s)",
            accept_multiple_files=True,
            type=["pdf"]
        )

        st.markdown("---")
        st.subheader("⚙️ Build Context")
        if st.button("Process", use_container_width=True):
            if not resume_pdfs:
                st.error("Please upload at least one resume PDF.")
            else:
                with st.spinner("Extracting & embedding…"):
                    # Resume text
                    resume_text = read_pdf_files(resume_pdfs)
                    st.session_state.resume_text = resume_text

                    # JD text (optional)
                    jd_text = read_pdf_files(jd_pdfs) if jd_pdfs else ""
                    st.session_state.jd_text = jd_text

                    # Build vector store over RESUME (and optionally JD to enrich retrieval)
                    corpus = resume_text
                    if jd_text.strip():
                        corpus += "\n\n---\n\nJOB DESCRIPTION(S):\n" + jd_text

                    chunks = chunk_text(corpus, chunk_size=300, chunk_overlap=50)
                    store = build_vectorstore(chunks)
                    st.session_state.conversation = make_conversation_chain(store)

                st.success("Context ready! Ask questions below.")

        if st.session_state.resume_text:
            st.markdown("---")
            st.subheader("🔎 Quick JD Match Check")
            if st.session_state.jd_text:
                # compute gaps
                rk, jk, missing = highlight_gaps(st.session_state.resume_text, st.session_state.jd_text, top_k=40)
                st.markdown("**Top resume keywords**: " + (", ".join(rk) if rk else "—"))
                st.markdown("**Top JD keywords**: " + (", ".join(jk) if jk else "—"))
                st.markdown("**Likely missing from resume**: " + (", ".join(missing) if missing else "Good coverage ✅"))
                if missing:
                    st.info(
                        "Tip: consider adding a bullet that demonstrates these skills/keywords if accurate. "
                        "Quantify with metrics (%, ×, time saved, throughput, etc.)."
                    )
            else:
                st.caption("Upload a job description on the left to see keyword coverage.")

    # ---------------- Main: Chat ----------------
    st.subheader("💬 Ask how to improve your resume")
    st.markdown(
        "> Examples:\n"
        "- How can I tailor this resume for data engineering?\n"
        "- Which bullets are too vague and how would you rewrite them?\n"
        "- What keywords am I missing for ML roles?\n"
        "- Suggest a stronger summary section based on my experience."
    )

    user_q = st.text_input("Your question:")
    if user_q:
        if not st.session_state.conversation:
            st.warning("Upload and click **Process** first.")
        else:
            with st.spinner("Thinking…"):
                response = st.session_state.conversation({"question": user_q})
                st.session_state.chat_history = response.get("chat_history", [])
            render_chat(st.session_state.chat_history)

    # ---------------- Footer info ----------------
    st.markdown("---")
    st.caption(
        "Built with Streamlit + LangChain (FAISS, OpenAI embeddings & chat). "
        "To deploy on Android, wrap this URL with Trusted Web Activity (TWA)."
    )


if __name__ == "__main__":
    main()
