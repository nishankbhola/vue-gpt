import os
import time
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions



# --- NEW: Cached function to load the embedding model ---
# @st.cache_resource
# def load_embedding_model():
#     """Loads the sentence transformer model only once."""
#     return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Helper to detect cloud
def is_streamlit_cloud():
    return os.environ.get("HOME") == "/home/adminuser"

def clean_vectorstore_directory(persist_directory):
    """Clean up vectorstore directory completely with better error handling"""
    if os.path.exists(persist_directory):
        try:
            # Force close any open database connections
            for root, dirs, files in os.walk(persist_directory):
                for file in files:
                    if file.endswith('.sqlite3') or file.endswith('.db'):
                        db_path = os.path.join(root, file)
                        try:
                            # Try to close any open connections
                            conn = sqlite3.connect(db_path)
                            conn.close()
                        except:
                            pass
            
            # Wait a bit for connections to close
            time.sleep(1)
            
            # Remove the directory
            shutil.rmtree(persist_directory)
            print(f"🧹 Cleaned up directory: {persist_directory}")
        except Exception as e:
            print(f"⚠️ Error cleaning directory: {e}")
            # If we can't remove it, try to remove just the db files
            try:
                for root, dirs, files in os.walk(persist_directory):
                    for file in files:
                        if file.endswith('.sqlite3') or file.endswith('.db'):
                            os.remove(os.path.join(root, file))
                print("🧹 Cleaned up database files")
            except:
                pass
    
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)

def ingest_company_pdfs(company_name: str, persist_directory: str = None):
    pdf_folder = os.path.join("data/pdfs", company_name)

    if persist_directory is None:
        base_path = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
        persist_directory = os.path.join(base_path, company_name)

    print("🗂️ Using vectorstore path:", persist_directory)

    if not os.path.exists(pdf_folder):
        raise ValueError(f"PDF folder not found: {pdf_folder}")
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_folder}")

    print(f"📄 Found {len(pdf_files)} PDF files")

    # Clean up old vectorstore
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    # Process PDFs
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for filename in pdf_files:
        print(f"📖 Processing: {filename}")
        file_path = os.path.join(pdf_folder, filename)
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if not pages:
                continue
                
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                length_function=len
            )
            chunks = splitter.split_documents(pages)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk.page_content)
                all_metadatas.append({
                    'source': filename,
                    'page': chunk.metadata.get('page', 0)
                })
                all_ids.append(f"{filename}_{i}")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    if not all_chunks:
        raise ValueError("No chunks were created from any PDF files")

    print(f"📊 Total chunks to process: {len(all_chunks)}")

    # Create ChromaDB client and collection
    client = chromadb.PersistentClient(path=persist_directory)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name=f"{company_name}_docs",
        embedding_function=embedding_function
    )
    
    # Add documents to collection
    collection.add(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    print(f"✅ Successfully created vectorstore for {company_name}")
    print(f"📈 Ingested {len(all_chunks)} chunks")
    
    return client, collection


if __name__ == "__main__":
    # Test function
    company_name = "test_company"
    ingest_company_pdfs(company_name)
