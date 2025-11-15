import os
import json
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------
# Document Loader
# ------------------------------------
def load_document(file_path: str):
    """Load documents from PDF, TXT, or DOCX."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            print(f"📘 Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            print(f"📄 Loading TXT: {file_path}")
            loader = TextLoader(file_path)
        elif ext == ".docx":
            print(f"📝 Loading DOCX: {file_path}")
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = loader.load()

        # Split into chunks for embedding
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        print(f"✅ Loaded {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"❌ Document loading failed: {e}")
        return []


# ------------------------------------
# Vector Store
# ------------------------------------
def create_vector_store(docs, persist_directory="chroma_db"):
    """Create a Chroma vector store with Ollama embeddings."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        os.makedirs(persist_directory, exist_ok=True)
        print("🧠 Creating or connecting to Chroma vector store...")

        # The new Chroma client doesn't require .persist()
        vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

        if docs:
            print(f"➕ Adding {len(docs)} documents to Chroma collection...")
            vector_store.add_documents(docs)

        # Custom history tracking
        vector_store.qa_history = []

        print("✅ Vector store ready.")
        return vector_store

    except Exception as e:
        print(f"❌ Vector store creation failed: {e}")
        return None


# ------------------------------------
# RAG Chain
# ------------------------------------
def create_rag_chain(vector_store):
    """Create a retrieval-based QA chain."""
    try:
        if vector_store is None:
            raise ValueError("Vector store not initialized")

        print("🧩 Building RetrievalQA chain...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = Ollama(model="llama3")

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            input_key="query",
            return_source_documents=True,
        )

        print("✅ RAG chain created successfully.")
        return chain
    except Exception as e:
        print(f"❌ RAG chain creation failed: {e}")
        return None


# ------------------------------------
# Store Q&A History
# ------------------------------------
def store_rag_answer(vector_store, answer: str, question: str, filename: str):
    """Save question-answer pairs in memory and optionally to disk."""
    try:
        if not hasattr(vector_store, "qa_history"):
            vector_store.qa_history = []
        vector_store.qa_history.append((question, answer))

        os.makedirs("temp_files", exist_ok=True)
        history_path = os.path.join("temp_files", f"{filename}_qa_history.json")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(vector_store.qa_history, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved Q&A history → {history_path}")
    except Exception as e:
        print(f"⚠️ Failed to store Q&A history: {e}")

