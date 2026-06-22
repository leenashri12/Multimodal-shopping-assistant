import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_PATH = os.path.join(DB_DIR, "store_info.txt")
INDEX_PATH = os.path.join(DB_DIR, "faiss_index")

def build_vector_db():
    if not os.path.exists(INFO_PATH):
        print(f"Error: {INFO_PATH} not found.")
        return

    print("Loading store knowledge base...")
    loader = TextLoader(INFO_PATH, encoding="utf-8")
    documents = loader.load()

    print("Splitting text into semantic chunks...")
    # Using 300 character chunks with 50 character overlap for fine-grained retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"Generated {len(docs)} document chunks.")

    print("Initializing HuggingFace Embedding Model ('all-MiniLM-L6-v2')...")
    # Using the local sentence-transformer model (automatically downloads on first run)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Generating embeddings and building FAISS vector database...")
    db = FAISS.from_documents(docs, embeddings)

    print(f"Saving FAISS index locally to: {INDEX_PATH}...")
    db.save_local(INDEX_PATH)
    print("Vector database initialization completed successfully!")

if __name__ == "__main__":
    build_vector_db()
