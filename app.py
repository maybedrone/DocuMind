import streamlit as st
import os
from rag import build_rag_chain

os.makedirs("uploads", exist_ok=True)

st.title("**DocuMind — Ask Your Documents**")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    save_path = os.path.join("uploads", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "chain" not in st.session_state:
        with st.spinner("Building RAG pipeline..."):
            st.session_state.chain, st.session_state.retriever = build_rag_chain(save_path)
        st.success("Ready! Ask a question.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask something about the PDF...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.chain.invoke(question)
                source_docs = st.session_state.retriever.invoke(question)
                sources = "\n".join([f"- Page {doc.metadata.get('page', '?')}" for doc in source_docs])
                full_answer = f"{answer}\n\n**Sources:**\n{sources}"
            except Exception as e:
                full_answer = "⚠️ Due to high demand, the AI service is temporarily unavailable. Please try again in a few minutes."
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
        st.rerun()
