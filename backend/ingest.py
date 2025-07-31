import os
import time
import shutil
import gc
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import pwd
import grp
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def force_cleanup_chromadb(persist_directory, max_attempts=3):
    """Aggressively clean up ChromaDB directory with multiple methods"""
    if not os.path.exists(persist_directory):
        return True
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"🧹 Cleanup attempt {attempt + 1} for: {persist_directory}")
            
            # Force garbage collection to release any handles
            gc.collect()
            time.sleep(1)
            
            # Method 1: Standard removal
            try:
                shutil.rmtree(persist_directory)
                logger.info("✅ Standard cleanup successful")
                time.sleep(1)
                return True
            except PermissionError:
                # Method 2: Change permissions and retry
                try:
                    for root, dirs, files in os.walk(persist_directory):
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                os.chmod(file_path, 0o777)
                            except:
                                pass
                        for dir in dirs:
                            try:
                                dir_path = os.path.join(root, dir)
                                os.chmod(dir_path, 0o777)
                            except:
                                pass
                    
                    shutil.rmtree(persist_directory)
                    logger.info("✅ Permission-based cleanup successful")
                    time.sleep(1)
                    return True
                except Exception as e2:
                    logger.warning(f"Permission-based cleanup failed: {e2}")
            except Exception as e1:
                logger.warning(f"Standard cleanup failed: {e1}")
            
            # Method 3: Sudo removal
            try:
                result = subprocess.run(['sudo', 'rm', '-rf', persist_directory], 
                                      check=True, capture_output=True, text=True)
                logger.info("✅ Sudo cleanup successful")
                time.sleep(1)
                return True
            except subprocess.CalledProcessError as e3:
                logger.warning(f"Sudo cleanup failed: {e3}")
            except Exception as e3:
                logger.warning(f"Sudo cleanup error: {e3}")
                
            # Wait before next attempt
            if attempt < max_attempts - 1:
                time.sleep(2 * (attempt + 1))
                
        except Exception as e:
            logger.error(f"Cleanup attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 * (attempt + 1))
    
    logger.error(f"❌ All cleanup attempts failed for: {persist_directory}")
    return False

def clean_vectorstore_directory(persist_directory):
    """Clean up vectorstore directory completely with better error handling"""
    logger.info(f"🧹 Cleaning vectorstore directory: {persist_directory}")
    
    # Force cleanup with multiple methods
    if not force_cleanup_chromadb(persist_directory):
        logger.warning("⚠️ Could not completely clean directory, continuing anyway...")
    
    # Ensure directory exists and is clean
    os.makedirs(persist_directory, exist_ok=True)
    set_www_data_ownership(persist_directory)
    
    logger.info(f"✅ Directory prepared: {persist_directory}")

def create_chromadb_client_with_retry(persist_directory, company_name, max_retries=3):
    """Create ChromaDB client with retry logic for version 1.0.15"""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔗 Creating ChromaDB client (attempt {attempt + 1})")
            
            # Force garbage collection
            gc.collect()
            time.sleep(1)
            
            # Create client
            client = chromadb.PersistentClient(path=persist_directory)
            
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Use a clean collection name
            collection_name = f"{company_name}_docs".replace("-", "_").replace(" ", "_")
            
            # Delete existing collection if it exists
            try:
                existing_collections = client.list_collections()
                for col in existing_collections:
                    if col.name == collection_name:
                        client.delete_collection(name=collection_name)
                        logger.info(f"🗑️ Deleted existing collection: {collection_name}")
                        time.sleep(1)  # Wait after deletion
                        break
            except Exception as e:
                logger.info(f"ℹ️ No existing collection found or error during deletion: {e}")
            
            # Create new collection
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"✨ Created new collection: {collection_name}")
            
            return client, collection
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                # Clean up and retry
                logger.info(f"🔄 Retrying in {2 * (attempt + 1)} seconds...")
                force_cleanup_chromadb(persist_directory)
                time.sleep(2 * (attempt + 1))
                os.makedirs(persist_directory, exist_ok=True)
                set_www_data_ownership(persist_directory)
            else:
                logger.error(f"❌ All attempts failed to create ChromaDB client")
                raise e

def ingest_company_pdfs(company_name: str, persist_directory: str = None):
    """Ingest PDFs for a company with enhanced error handling and cleanup"""
    
    pdf_folder = os.path.join("data/pdfs", company_name)

    if persist_directory is None:
        base_path = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
        persist_directory = os.path.join(base_path, company_name)

    logger.info(f"🗂️ Using vectorstore path: {persist_directory}")
    logger.info(f"📁 PDF folder: {pdf_folder}")

    if not os.path.exists(pdf_folder):
        raise ValueError(f"PDF folder not found: {pdf_folder}")
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_folder}")

    logger.info(f"📄 Found {len(pdf_files)} PDF files: {pdf_files}")

    # Clean up old vectorstore completely
    clean_vectorstore_directory(persist_directory)

    # Process PDFs
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for filename in pdf_files:
        logger.info(f"📖 Processing: {filename}")
        file_path = os.path.join(pdf_folder, filename)
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if not pages:
                logger.warning(f"⚠️ No pages found in {filename}")
                continue
                
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                length_function=len
            )
            chunks = splitter.split_documents(pages)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created from {filename}")
                continue
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk.page_content)
                all_metadatas.append({
                    'source': filename,
                    'page': chunk.metadata.get('page', 0)
                })
                all_ids.append(f"{filename}_{i}")
            
            logger.info(f"✅ Processed {filename}: {len(chunks)} chunks")
                
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
            continue

    if not all_chunks:
        raise ValueError("No chunks were created from any PDF files")

    logger.info(f"📊 Total chunks to process: {len(all_chunks)}")

    try:
        # Create ChromaDB client with retry
        client, collection = create_chromadb_client_with_retry(persist_directory, company_name)
        
        # Add documents to collection in smaller batches for stability
        batch_size = 25  # Even smaller batches for better stability
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_chunks), batch_size):
            batch_num = i // batch_size + 1
            batch_chunks = all_chunks[i:i+batch_size]
            batch_metadatas = all_metadatas[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            
            try:
                logger.info(f"📝 Adding batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                
                collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                logger.info(f"✅ Successfully added batch {batch_num}/{total_batches}")
                
                # Small delay between batches for stability
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"❌ Error adding batch {batch_num}: {e}")
                raise e
        
        # Verify the collection has data
        try:
            count = collection.count()
            logger.info(f"📈 Collection verification: {count} documents in collection")
            if count == 0:
                raise ValueError("No documents were successfully added to the collection")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify collection count: {e}")
        
        # Set proper ownership for all created files
        try:
            logger.info("🔐 Setting proper file ownership...")
            for root, dirs, files in os.walk(persist_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    set_www_data_ownership(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    set_www_data_ownership(dir_path)
            logger.info("✅ File ownership set successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not set ownership for all files: {e}")
        
        logger.info(f"🎉 Successfully created vectorstore for {company_name}")
        logger.info(f"📈 Ingested {len(all_chunks)} chunks from {len(pdf_files)} PDF files")
        
        return client, collection
        
    except Exception as e:
        logger.error(f"❌ Critical error during vectorstore creation: {e}")
        
        # Clean up on failure
        try:
            force_cleanup_chromadb(persist_directory)
        except:
            pass
            
        raise e


if __name__ == "__main__":
    # Test function
    import sys
    if len(sys.argv) > 1:
        company_name = sys.argv[1]
    else:
        company_name = "test_company"
    
    try:
        logger.info(f"🧪 Testing ingestion for company: {company_name}")
        client, collection = ingest_company_pdfs(company_name)
        logger.info("✅ Test completed successfully")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)
