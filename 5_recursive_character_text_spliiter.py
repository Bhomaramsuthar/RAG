import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# 1. Imported the newer Recursive splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading all the documents from {docs_path}. . . .")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )

    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    for i,doc in enumerate(documents[:2]): 
        print(f"\nDocument {i+1}:")
        print(f"    Source: {doc.metadata['source']}")
        print(f"    Content length: {len(doc.page_content)} characters")
        print(f"    Content preview: {doc.page_content[:100]}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print("\nSplitting the document into chunks . . .")

    # 2. Replaced the old splitter with the Recursive version
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content}")
            print("-" * 50)
       
        if len(chunks) > 5:
            print(f"\n. . . and {len(chunks)-5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    
    # Using the free local embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("\n--- Creating the Vector DB ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    
    print(f"Vector DB created and saved to {persist_directory}")
    return vectorstore

def main():
    print("----- RAG Document Ingestion Pipeline -----\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):
        print("Vector store already exists. No need to re-process documents.")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space":"cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore
    
    print("Persistent directory does not exist. Initializing vector store . . .\n")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\nIngestion complete !!!")
    return vectorstore

if __name__ == "__main__":
    main()