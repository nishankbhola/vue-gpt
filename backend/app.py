import os
import shutil
import json
import requests
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import chromadb
from chromadb.utils import embedding_functions
import logging
from functools import lru_cache
from flask import Flask, send_from_directory
from flask import Flask, request, jsonify, send_file, send_from_directory
import subprocess
import pwd
import grp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Add file upload size limit (500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB in bytes

# Configuration
UPLOAD_FOLDER = 'data/pdfs'
LOGOS_FOLDER = 'data/logos'
ALLOWED_EXTENSIONS = {'pdf'}
ADMIN_PASSWORD = "classmate"

# Gemini model fallback configuration
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro"
]

# Global state for model tracking
current_model_index = 0
vectorstore_cache = {}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGOS_FOLDER, exist_ok=True)

@lru_cache(maxsize=1)
def get_embedding_function():
    """Get ChromaDB embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def is_streamlit_cloud():
    return os.environ.get("HOME") == "/home/adminuser"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_vectorstore_root():
    """Get the vectorstore root directory"""
    return "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"

def reset_entire_chromadb():
    """Completely reset the entire ChromaDB database"""
    try:
        VECTORSTORE_ROOT = get_vectorstore_root()
        
        # Clear all cached vectorstores
        global vectorstore_cache
        vectorstore_cache.clear()
        
        # Force garbage collection to release any ChromaDB connections
        import gc
        gc.collect()
        
        # Wait longer for ChromaDB 1.0.15
        time.sleep(3)
        
        # Remove the entire vectorstore directory
        if os.path.exists(VECTORSTORE_ROOT):
            logger.info(f"🗑️ Removing entire vectorstore directory: {VECTORSTORE_ROOT}")
            
            # Try multiple methods to ensure complete removal
            try:
                # Method 1: Standard removal
                shutil.rmtree(VECTORSTORE_ROOT)
            except Exception as e1:
                logger.warning(f"Standard removal failed: {e1}")
                try:
                    # Method 2: Force removal with sudo
                    subprocess.run(['sudo', 'rm', '-rf', VECTORSTORE_ROOT], 
                                 check=True, capture_output=True)
                except Exception as e2:
                    logger.warning(f"Sudo removal failed: {e2}")
                    # Method 3: Manual file-by-file removal
                    try:
                        for root, dirs, files in os.walk(VECTORSTORE_ROOT, topdown=False):
                            for file in files:
                                try:
                                    os.chmod(os.path.join(root, file), 0o777)
                                    os.remove(os.path.join(root, file))
                                except:
                                    pass
                            for dir in dirs:
                                try:
                                    os.chmod(os.path.join(root, dir), 0o777)
                                    os.rmdir(os.path.join(root, dir))
                                except:
                                    pass
                        os.rmdir(VECTORSTORE_ROOT)
                    except Exception as e3:
                        logger.error(f"Manual removal also failed: {e3}")
                        raise e3
        
        # Wait longer for filesystem to sync with ChromaDB 1.0.15
        time.sleep(2)
        
        # Recreate the base directory
        os.makedirs(VECTORSTORE_ROOT, exist_ok=True)
        set_www_data_ownership(VECTORSTORE_ROOT)
        
        logger.info("✅ ChromaDB database completely reset")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to reset ChromaDB: {e}")
        return False

def create_chroma_vectorstore(vectorstore_path, company_name, max_retries=5):
    """Create ChromaDB client with enhanced retry logic for version 1.0.15"""
    for attempt in range(max_retries):
        try:
            if company_name in vectorstore_cache:
                del vectorstore_cache[company_name]
            
            # Ensure the directory exists and is clean
            if os.path.exists(vectorstore_path):
                shutil.rmtree(vectorstore_path)
                time.sleep(2)  # Longer wait for 1.0.15
            os.makedirs(vectorstore_path, exist_ok=True)
            
            # Set proper ownership for vectorstore directory
            set_www_data_ownership(vectorstore_path)
            
            # Create ChromaDB client for version 1.0.15
            client = chromadb.PersistentClient(path=vectorstore_path)
            
            embedding_function = get_embedding_function()
            
            # Use a simpler collection name
            collection_name = f"{company_name}_docs".replace("-", "_").replace(" ", "_").lower()
            
            # Try to delete existing collection first
            try:
                existing_collections = client.list_collections()
                for col in existing_collections:
                    if col.name == collection_name:
                        client.delete_collection(name=collection_name)
                        time.sleep(2)  # Longer wait after deletion
                        break
            except Exception:
                pass  # Collection might not exist
            
            # Create new collection with get_or_create
            try:
                collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
            except Exception:
                # Fallback to create_collection
                collection = client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
            
            return client, collection
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)  # Longer wait times
                time.sleep(wait_time)
                
                if os.path.exists(vectorstore_path):
                    try:
                        shutil.rmtree(vectorstore_path)
                    except:
                        pass
            else:
                raise e

def get_company_vectorstore(company_name, vectorstore_path):
    """Get or create company-specific vectorstore with proper caching for ChromaDB 1.0.15"""
    if company_name not in vectorstore_cache:
        try:
            # Simple client creation for version 1.0.15
            client = chromadb.PersistentClient(path=vectorstore_path)
            embedding_function = get_embedding_function()
            
            # Use consistent collection name
            collection_name = f"{company_name}_docs".replace("-", "_").replace(" ", "_").lower()
            
            try:
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
            except Exception as e:
                logger.error(f"Error getting collection {collection_name}: {e}")
                raise ValueError(f"Collection {collection_name} not found. Please click 'Relearn PDFs' first.")
            
            vectorstore_cache[company_name] = (client, collection)
        except Exception as e:
            logger.error(f"Error creating vectorstore client: {e}")
            raise e
    return vectorstore_cache[company_name]
    

def clear_company_vectorstore_cache(company_name):
    """Clear vectorstore cache for a specific company"""
    if company_name in vectorstore_cache:
        del vectorstore_cache[company_name]

def call_gemini_with_fallback(payload):
    """Call Gemini API with automatic model fallback on rate limit"""
    global current_model_index
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(len(GEMINI_MODELS)):
        current_model = GEMINI_MODELS[current_model_index]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={GEMINI_API_KEY}"
         
        try:
            time.sleep(0.5)
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                return response, current_model
            elif response.status_code == 429:
                logger.warning(f"Rate limit reached for {current_model}, trying next model...")
                current_model_index = (current_model_index + 1) % len(GEMINI_MODELS)
                time.sleep(2)
                continue
            else:
                return response, current_model
                
        except Exception as e:
            logger.error(f"Error with {current_model}: {str(e)}")
            current_model_index = (current_model_index + 1) % len(GEMINI_MODELS)
            continue
    
    return response, current_model

# API Routes

@app.route('/api/auth/admin', methods=['POST'])
def admin_auth():
    """Admin authentication"""
    data = request.get_json()
    password = data.get('password', '')
    
    if password == ADMIN_PASSWORD:
        return jsonify({'success': True, 'message': 'Admin access granted'})
    else:
        return jsonify({'success': False, 'message': 'Invalid password'}), 401

@app.route('/api/companies', methods=['GET'])
def get_companies():
    """Get list of all companies"""
    try:
        # Check if upload folder exists, if not create it
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
        company_folders = [f for f in os.listdir(UPLOAD_FOLDER) 
                          if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
        
        companies = []
        for company in company_folders:
            # Check if logo exists
            logo_path = os.path.join(LOGOS_FOLDER, f"{company}.png")
            has_logo = os.path.exists(logo_path)
            
            # Get PDF count
            company_pdf_dir = os.path.join(UPLOAD_FOLDER, company)
            try:
                pdf_count = len([f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")])
            except:
                pdf_count = 0
            
            # Check if vectorstore exists
            VECTORSTORE_ROOT = get_vectorstore_root()
            vectorstore_path = os.path.join(VECTORSTORE_ROOT, company)
            has_vectorstore = os.path.exists(vectorstore_path)
            
            companies.append({
                'name': company,
                'has_logo': has_logo,
                'pdf_count': pdf_count,
                'has_vectorstore': has_vectorstore
            })
        
        return jsonify({'companies': companies})
    except Exception as e:
        logger.error(f"Error getting companies: {str(e)}")
        return jsonify({'companies': [], 'error': str(e)}), 200  # Return empty list instead of error

def set_www_data_ownership(file_or_dir_path):
    """Set ownership to www-data:www-data for uploaded files/directories"""
    try:
        # Get www-data user and group IDs
        www_data_user = pwd.getpwnam('www-data')
        www_data_group = grp.getgrnam('www-data')
        
        # Change ownership
        os.chown(file_or_dir_path, www_data_user.pw_uid, www_data_group.gr_gid)
        
        # Set appropriate permissions
        if os.path.isdir(file_or_dir_path):
            os.chmod(file_or_dir_path, 0o755)  # rwxr-xr-x for directories
        else:
            os.chmod(file_or_dir_path, 0o644)  # rw-r--r-- for files
        
        logger.info(f"Set www-data ownership for: {file_or_dir_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Could not set www-data ownership for {file_or_dir_path}: {e}")
        # Try with sudo as fallback
        try:
            if os.path.isdir(file_or_dir_path):
                subprocess.run(['sudo', 'chown', '-R', 'www-data:www-data', file_or_dir_path], 
                             check=True, capture_output=True)
                subprocess.run(['sudo', 'chmod', '-R', '755', file_or_dir_path], 
                             check=True, capture_output=True)
            else:
                subprocess.run(['sudo', 'chown', 'www-data:www-data', file_or_dir_path], 
                             check=True, capture_output=True)
                subprocess.run(['sudo', 'chmod', '644', file_or_dir_path], 
                             check=True, capture_output=True)
            logger.info(f"Set www-data ownership with sudo for: {file_or_dir_path}")
            return True
        except Exception as sudo_error:
            logger.error(f"Failed to set ownership even with sudo: {sudo_error}")
            return False

@app.route('/api/companies', methods=['POST'])
def add_company():
    """Add a new company"""
    try:
        company_name = request.form.get('company_name')
        if not company_name:
            return jsonify({'error': 'Company name is required'}), 400
        
        # Create company directory
        company_path = os.path.join(UPLOAD_FOLDER, company_name)
        if os.path.exists(company_path):
            return jsonify({'error': 'Company already exists'}), 400
        
        os.makedirs(company_path)
        
        # Set proper ownership for the new directory
        set_www_data_ownership(company_path)
        
        # Handle logo upload
        if 'logo' in request.files:
            logo_file = request.files['logo']
            if logo_file.filename != '':
                logo_path = os.path.join(LOGOS_FOLDER, f"{company_name}.png")
                logo_file.save(logo_path)
                
                # Set proper ownership for the logo file
                set_www_data_ownership(logo_path)
        
        return jsonify({'success': True, 'message': f'Added company: {company_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/companies/<company_name>', methods=['DELETE'])
def delete_company(company_name):
    """Delete a company and all its data with ChromaDB reset option"""
    import stat
    import time
    
    def remove_readonly(func, path, _):
        """Error handler for removing read-only files on Windows"""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            logger.warning(f"Could not remove {path}: {e}")
    
    def safe_remove_tree(path, max_retries=3):
        """Safely remove directory tree with retries"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(path):
                    # First try normal removal
                    shutil.rmtree(path)
                    return True
            except PermissionError:
                try:
                    # Try with error handler for read-only files
                    shutil.rmtree(path, onerror=remove_readonly)
                    return True
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed to remove {path}: {e}. Retrying...")
                        time.sleep(1)  # Wait 1 second before retry
                    else:
                        logger.error(f"Failed to remove {path} after {max_retries} attempts: {e}")
                        return False
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed to remove {path}: {e}. Retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to remove {path} after {max_retries} attempts: {e}")
                    return False
        return False
    
    try:
        # Clear vectorstore cache first
        clear_company_vectorstore_cache(company_name)
        
        # Check if this is the last company - if so, reset entire ChromaDB
        company_folders = [f for f in os.listdir(UPLOAD_FOLDER) 
                          if os.path.isdir(os.path.join(UPLOAD_FOLDER, f)) and f != company_name]
        
        errors = []
        
        # Delete PDFs
        company_path = os.path.join(UPLOAD_FOLDER, company_name)
        if os.path.exists(company_path):
            if not safe_remove_tree(company_path):
                errors.append(f"Could not fully remove PDF directory: {company_path}")
        
        # Delete vectorstore
        VECTORSTORE_ROOT = get_vectorstore_root()
        vectorstore_path = os.path.join(VECTORSTORE_ROOT, company_name)
        if os.path.exists(vectorstore_path):
            if not safe_remove_tree(vectorstore_path):
                errors.append(f"Could not fully remove vectorstore: {vectorstore_path}")
        
        # Delete logo
        logo_path = os.path.join(LOGOS_FOLDER, f"{company_name}.png")
        if os.path.exists(logo_path):
            try:
                os.chmod(logo_path, stat.S_IWRITE)  # Remove read-only if needed
                os.remove(logo_path)
            except Exception as e:
                errors.append(f"Could not remove logo: {e}")
        
        # If no companies left or if there were vectorstore errors, reset entire ChromaDB
        if len(company_folders) == 0 or any("vectorstore" in error for error in errors):
            logger.info("🔄 Resetting entire ChromaDB database...")
            if reset_entire_chromadb():
                logger.info("✅ ChromaDB database reset successfully")
                # Clear any vectorstore-related errors since we reset everything
                errors = [error for error in errors if "vectorstore" not in error.lower()]
            else:
                errors.append("Failed to reset ChromaDB database")
        
        if errors:
            # Some files couldn't be deleted, but we'll return partial success
            error_message = "Company deleted with some issues: " + "; ".join(errors)
            return jsonify({
                'success': True, 
                'message': f'Deleted {company_name}', 
                'warnings': errors
            }), 200
        else:
            return jsonify({'success': True, 'message': f'Deleted all data for {company_name}'})
            
    except Exception as e:
        logger.error(f"Error deleting company {company_name}: {e}")
        # Try to reset ChromaDB as a last resort
        try:
            reset_entire_chromadb()
            return jsonify({
                'success': True, 
                'message': f'Deleted {company_name} (with database reset)', 
                'warnings': [f'Had to reset database due to error: {str(e)}']
            }), 200
        except:
            return jsonify({'error': f'Error deleting company: {str(e)}'}), 500

# Add new endpoint for manual ChromaDB reset
@app.route('/api/reset-database', methods=['POST'])
def reset_database():
    """Manually reset the entire ChromaDB database"""
    try:
        data = request.get_json()
        password = data.get('password', '')
        
        # Require admin password for this operation
        if password != ADMIN_PASSWORD:
            return jsonify({'error': 'Admin password required'}), 401
        
        logger.info("🔄 Manual ChromaDB reset requested...")
        
        if reset_entire_chromadb():
            return jsonify({
                'success': True, 
                'message': 'ChromaDB database reset successfully. All companies will need to relearn their PDFs.'
            })
        else:
            return jsonify({'error': 'Failed to reset ChromaDB database'}), 500
            
    except Exception as e:
        logger.error(f"Error in manual database reset: {e}")
        return jsonify({'error': f'Error resetting database: {str(e)}'}), 500

@app.route('/api/companies/<company_name>/logo', methods=['GET'])
def get_company_logo(company_name):
    """Get company logo"""
    logo_path = os.path.join(LOGOS_FOLDER, f"{company_name}.png")
    if os.path.exists(logo_path):
        return send_file(logo_path, mimetype='image/png')
    return jsonify({'error': 'Logo not found'}), 404

@app.route('/api/companies/<company_name>/pdfs', methods=['GET'])
def get_company_pdfs(company_name):
    """Get list of PDFs for a company"""
    try:
        company_pdf_dir = os.path.join(UPLOAD_FOLDER, company_name)
        if not os.path.exists(company_pdf_dir):
            return jsonify({'pdfs': []})
        
        pdf_files = [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
        
        pdfs = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(company_pdf_dir, pdf_file)
            try:
                file_size = os.path.getsize(pdf_path)
                size_mb = round(file_size / (1024 * 1024), 2)
            except:
                size_mb = 0
            
            pdfs.append({
                'name': pdf_file,
                'size_mb': size_mb
            })
        
        return jsonify({'pdfs': pdfs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/companies/<company_name>/pdfs', methods=['POST'])
def upload_pdf(company_name):
    """Upload PDF to company"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400
        
        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(pdf_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(pdf_file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, company_name, filename)
        
        # Create directory if it doesn't exist
        company_dir = os.path.dirname(save_path)
        if not os.path.exists(company_dir):
            os.makedirs(company_dir, exist_ok=True)
            # Set proper ownership for the directory
            set_www_data_ownership(company_dir)
        
        # Save the PDF file
        pdf_file.save(save_path)
        
        # Set proper ownership for the uploaded PDF
        set_www_data_ownership(save_path)
        
        return jsonify({'success': True, 'message': f'Uploaded: {filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/companies/<company_name>/pdfs/<pdf_name>', methods=['GET'])
def download_pdf(company_name, pdf_name):
    """Download PDF file"""
    try:
        pdf_path = os.path.join(UPLOAD_FOLDER, company_name, pdf_name)
        if os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True, download_name=pdf_name)
        return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/companies/<company_name>/relearn', methods=['POST'])
def relearn_pdfs(company_name):
    """Rebuild knowledge base for company - Updated for ChromaDB 1.0.15 with better error handling"""
    try:
        from ingest import ingest_company_pdfs
        
        VECTORSTORE_ROOT = get_vectorstore_root()
        vectorstore_path = os.path.join(VECTORSTORE_ROOT, company_name)
        
        # Clear the cached vectorstore
        clear_company_vectorstore_cache(company_name)
        
        # Ensure base vectorstore directory exists
        os.makedirs(VECTORSTORE_ROOT, exist_ok=True)
        
        # Run the ingestion with better error handling
        try:
            vectordb = ingest_company_pdfs(company_name, persist_directory=vectorstore_path)
            return jsonify({'success': True, 'message': 'Knowledge base updated successfully!'})
        except ValueError as ve:
            # Handle specific PDF/document errors
            return jsonify({'error': f'Document processing error: {str(ve)}'}), 400
        except Exception as ce:
            # Handle ChromaDB specific errors
            clear_company_vectorstore_cache(company_name)
            
            # Check for common ChromaDB corruption indicators
            if any(indicator in str(ce).lower() for indicator in [
                "database is locked", "database corruption", "no such table", 
                "tenant", "sqlite", "disk i/o error", "malformed"
            ]):
                logger.warning(f"ChromaDB corruption detected for {company_name}: {ce}")
                
                # Try to reset the entire ChromaDB and retry once
                if reset_entire_chromadb():
                    try:
                        time.sleep(2)  # Wait for reset to complete
                        vectordb = ingest_company_pdfs(company_name, persist_directory=vectorstore_path)
                        return jsonify({
                            'success': True, 
                            'message': 'Knowledge base updated successfully after database reset!'
                        })
                    except Exception as retry_error:
                        return jsonify({
                            'error': f'Database corruption detected and reset attempted, but retry failed: {str(retry_error)}'
                        }), 500
                else:
                    return jsonify({
                        'error': 'Database corruption detected. Please try again or restart the server.'
                    }), 500
            else:
                # Other ChromaDB errors
                if "already exists" in str(ce).lower():
                    return jsonify({'error': 'Collection already exists. Please try again.'}), 500
                else:
                    return jsonify({'error': f'Vector database error: {str(ce)}'}), 500
        
    except ImportError as ie:
        return jsonify({'error': f'Import error: {str(ie)}'}), 500
    except Exception as e:
        clear_company_vectorstore_cache(company_name)
        logger.error(f"Unexpected error in relearn_pdfs: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
            
@app.route('/api/companies/<company_name>/ask', methods=['POST'])
def ask_company_question(company_name):
    """Ask a question to a specific company"""
    try:
        data = request.get_json()
        query = data.get('question', '')
        
        if not query:
            return jsonify({'error': 'Question is required'}), 400
        
        VECTORSTORE_ROOT = get_vectorstore_root()
        vectorstore_path = os.path.join(VECTORSTORE_ROOT, company_name)
        
        if not os.path.exists(vectorstore_path):
            return jsonify({'error': 'Knowledge base not found. Please upload PDFs and click Relearn.'}), 404
        
        # Get company-specific vectorstore
        client, collection = get_company_vectorstore(company_name, vectorstore_path)
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        # Convert results to document format
        docs = []
        for i, (text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            docs.append({
                'page_content': text,
                'metadata': metadata or {}
            })
        context = "\n\n".join([doc['page_content'] for doc in docs])

        payload = {
            "contents": [{
                "parts": [{
                    "text": f"""As a professional insurance broker assistant, answer the following question using ONLY the context provided for {company_name}.

Question: {query}

Context from {company_name}: {context}

Please provide a clear, professional response that would be helpful for insurance brokers and their clients. Base your answer ONLY on the provided context from {company_name}.
"""
                }]
            }]
        }

        response, used_model = call_gemini_with_fallback(payload)

        if response.status_code == 429:
            return jsonify({'error': 'Rate limit reached. Please wait a moment and try again.'}), 429
        elif response.status_code == 200:
            try:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                
                # Prepare source documents
                sources = []
                for i, doc in enumerate(docs[:3]):
                    source_info = {
                        'content': doc['page_content'][:500] + "..." if len(doc['page_content']) > 500 else doc['page_content'],
                        'source': doc['metadata'].get('source', 'Unknown'),
                        'index': i
                    }
                    sources.append(source_info)
                
                return jsonify({
                    'success': True,
                    'answer': answer,
                    'model_used': used_model,
                    'sources': sources,
                    'company': company_name,
                    'question': query
                })
                
            except Exception as e:
                return jsonify({'error': 'Error parsing response from Gemini'}), 500
        else:
            return jsonify({'error': f'Gemini API Error: {response.status_code}'}), 500
            
    except Exception as e:
        error_msg = str(e)
        if any(indicator in error_msg.lower() for indicator in [
            "no such table: tenants", "database corruption", "disk i/o error", "malformed"
        ]):
            clear_company_vectorstore_cache(company_name)
            return jsonify({'error': 'Database corruption detected. Please click Relearn PDFs to rebuild the knowledge base.'}), 500
        else:
            clear_company_vectorstore_cache(company_name)
            return jsonify({'error': f'Error accessing knowledge base: {error_msg}'}), 500

@app.route('/api/ask-all', methods=['POST'])
def ask_all_companies():
    """Ask a question to all companies"""
    try:
        data = request.get_json()
        query = data.get('question', '')
        
        if not query:
            return jsonify({'error': 'Question is required'}), 400
        
        company_folders = [f for f in os.listdir(UPLOAD_FOLDER) 
                          if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
        
        if not company_folders:
            return jsonify({'error': 'No companies found to query.'}), 404
        
        VECTORSTORE_ROOT = get_vectorstore_root()
        responses = []
        
        for company in company_folders:
            vectorstore_path = os.path.join(VECTORSTORE_ROOT, company)
            
            if os.path.exists(vectorstore_path):
                try:
                    client, collection = get_company_vectorstore(company, vectorstore_path)
                    results = collection.query(
                        query_texts=[query],
                        n_results=5
                    )
                    # Convert results to document format
                    docs = []
                    for i, (text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                        docs.append({
                            'page_content': text,
                            'metadata': metadata or {}
                        })
                    context = "\n\n".join([doc['page_content'] for doc in docs])
                    
                    
                    payload = {
                        "contents": [{
                            "parts": [{
                                "text": f"""As a professional insurance broker assistant, answer the following question using ONLY the context provided for {company}.

Question: {query}

Context from {company}: {context}

Please provide a clear, professional response that would be helpful for insurance brokers and their clients. Base your answer ONLY on the provided context from {company}.
"""
                            }]
                        }]
                    }

                    response, used_model = call_gemini_with_fallback(payload)

                    if response.status_code == 200:
                        try:
                            answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                            
                            # Prepare source documents
                            sources = []
                            for i, doc in enumerate(docs[:3]):
                                source_info = {
                                    'content': doc['page_content'][:500] + "..." if len(doc['page_content']) > 500 else doc['page_content'],
                                    'source': doc['metadata'].get('source', 'Unknown'),
                                    'index': i
                                }
                                sources.append(source_info)
                            
                            responses.append({
                                'company': company,
                                'answer': answer,
                                'model_used': used_model,
                                'sources': sources,
                                'success': True
                            })
                            
                        except Exception as e:
                            responses.append({
                                'company': company,
                                'error': 'Error parsing response from Gemini',
                                'success': False
                            })
                    else:
                        responses.append({
                            'company': company,
                            'error': f'Gemini API Error: {response.status_code}',
                            'success': False
                        })
                        
                except Exception as e:
                    error_msg = str(e)
                    if any(indicator in error_msg.lower() for indicator in [
                        "no such table: tenants", "database corruption", "disk i/o error", "malformed"
                    ]):
                        clear_company_vectorstore_cache(company)
                        responses.append({
                            'company': company,
                            'error': 'Database corruption detected. Please click Relearn PDFs to rebuild the knowledge base.',
                            'success': False
                        })
                    else:
                        clear_company_vectorstore_cache(company)
                        responses.append({
                            'company': company,
                            'error': f'Error accessing knowledge base: {error_msg}',
                            'success': False
                        })
            else:
                responses.append({
                    'company': company,
                    'error': 'Knowledge base not found.',
                    'success': False
                })
        
        return jsonify({
            'success': True,
            'question': query,
            'responses': responses
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources', methods=['GET'])
def get_all_resources():
    """Get all PDFs from all companies for resources view"""
    try:
        company_folders = [f for f in os.listdir(UPLOAD_FOLDER) 
                          if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
        
        all_resources = []
        total_pdfs = 0
        total_size = 0
        
        for company in company_folders:
            company_pdf_dir = os.path.join(UPLOAD_FOLDER, company)
            pdf_files = [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
            
            company_pdfs = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(company_pdf_dir, pdf_file)
                try:
                    file_size = os.path.getsize(pdf_path)
                    size_mb = round(file_size / (1024 * 1024), 2)
                    total_size += file_size
                except:
                    size_mb = 0
                
                company_pdfs.append({
                    'name': pdf_file,
                    'size_mb': size_mb
                })
            
            total_pdfs += len(pdf_files)
            
            # Check if logo exists
            logo_path = os.path.join(LOGOS_FOLDER, f"{company}.png")
            has_logo = os.path.exists(logo_path)
            
            all_resources.append({
                'company': company,
                'pdfs': company_pdfs,
                'has_logo': has_logo
            })
        
        total_size_mb = round(total_size / (1024 * 1024), 2)
        
        return jsonify({
            'resources': all_resources,
            'summary': {
                'total_companies': len(company_folders),
                'total_pdfs': total_pdfs,
                'total_size_mb': total_size_mb
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'message': 'Server is running'})


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files or return index.html for client-side routing"""
    try:
        return send_from_directory(app.static_folder, path)
    except:
        # Return index.html for client-side routing
        return send_from_directory(app.static_folder, 'index.html')

# Add at the very end of app.py, replace the existing if __name__ == '__main__':
if __name__ == '__main__':
    # For production, don't run debug mode
    debug_mode = False
    print("🚀 Starting Broker-GPT Backend Server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🖼️ Logos folder: {LOGOS_FOLDER}")
    print(f"🔧 Debug mode: {debug_mode}")
    print("🌐 Server will be available at: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
