import os
import time
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import pwd
import grp
import subprocess


# --- NEW: Cached function to load the embedding model ---
# @st.cache_resource
# def load_embedding_model():
#     """Loads the sentence transformer model only once."""
#     return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



# Add this function to ingest.py
def set_www_data_ownership(file_or_dir_path):
    """Set ownership to www-data:www-data for created files/directories"""
    try:
        www_data_user = pwd.getpwnam('www-data')
        www_data_group = grp.getgrnam('www-data')
        os.chown(file_or_dir_path, www_data_user.pw_uid, www_data_group.gr_gid)
        
        if os.path.isdir(file_or_dir_path):
            os.chmod(file_or_dir_path, 0o755)
        else:
            os.chmod(file_or_dir_path, 0o644)
        return True
    except Exception:
        try:
            if os.path.isdir(file_or_dir_path):
                subprocess.run(['sudo', 'chown', '-R', 'www-data:www-data', file_or_dir_path], check=True, capture_output=True)
                subprocess.run(['sudo', 'chmod', '-R', '755', file_or_dir_path], check=True, capture_output=True)
            else:
                subprocess.run(['sudo', 'chown', 'www-data:www-data', file_or_dir_path], check=True, capture_output=True)
                subprocess.run(['sudo', 'chmod', '644', file_or_dir_path], check=True, capture_output=True)
            return True
        except Exception:
            return False
            
# Helper to detect cloud
def is_streamlit_cloud():
    return os.environ.get("HOME") == "/home/adminuser"

def clean_vectorstore_directory(persist_directory):
    """Clean up vectorstore directory completely with better error handling"""
    if os.path.exists(persist_directory):
        try:
            # Wait a bit for any processes to finish
            time.sleep(1)
            
            # Remove the directory
            shutil.rmtree(persist_directory)
            print(f"🧹 Cleaned up directory: {persist_directory}")
        except Exception as e:
            print(f"⚠️ Error cleaning directory: {e}")
    
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

    # Clean up old vectorstore completely
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        time.sleep(1)  # Wait for cleanup
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

    # Create ChromaDB client and collection with proper settings
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Use a clean collection name
    collection_name = f"{company_name}_docs".replace("-", "_").replace(" ", "_")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection might not exist
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    # Add documents to collection in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        
        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"📝 Added batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
    
    # Set proper ownership for all created files
    try:
        for root, dirs, files in os.walk(persist_directory):
            for file in files:
                file_path = os.path.join(root, file)
                set_www_data_ownership(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                set_www_data_ownership(dir_path)
    except Exception as e:
        print(f"⚠️ Could not set ownership for all files: {e}")
    
    print(f"✅ Successfully created vectorstore for {company_name}")
    print(f"📈 Ingested {len(all_chunks)} chunks")
    
    return client, collection


if __name__ == "__main__":
    # Test function
    company_name = "test_company"
    ingest_company_pdfs(company_name)
