import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

def build_rag_chain(pdf_path: str):
    # Load
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # Embed + Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(docs, embeddings)

    # Retriever
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Prompt
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided context. Be direct, no filler phrases like "based on the transcript".
If the context is insufficient, say you don't know.

{context}
Question: {question}
        """,
        input_variables=['context', 'question']
    )

    # Model
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="conversational",
        provider="novita"
    )
    model = ChatHuggingFace(llm=llm)

    # Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    }) | prompt | model | StrOutputParser()

    return chain, retriever