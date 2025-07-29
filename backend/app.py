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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

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

def create_chroma_vectorstore(vectorstore_path, company_name, max_retries=5):
    """Create ChromaDB client with enhanced retry logic"""
    for attempt in range(max_retries):
        try:
            if company_name in vectorstore_cache:
                del vectorstore_cache[company_name]
            
            os.makedirs(vectorstore_path, exist_ok=True)
            
            client = chromadb.PersistentClient(path=vectorstore_path)
            embedding_function = get_embedding_function()
            
            # Get or create collection
            collection = client.get_or_create_collection(
                name=f"{company_name}_docs",
                embedding_function=embedding_function
            )
            
            return client, collection
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                time.sleep(wait_time)
                
                if os.path.exists(vectorstore_path):
                    try:
                        shutil.rmtree(vectorstore_path)
                    except:
                        pass
            else:
                raise e

def get_company_vectorstore(company_name, vectorstore_path):
    """Get or create company-specific vectorstore with proper caching"""
    if company_name not in vectorstore_cache:
        client, collection = create_chroma_vectorstore(vectorstore_path, company_name)
        vectorstore_cache[company_name] = (client, collection)
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
            VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
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
        
        # Handle logo upload
        if 'logo' in request.files:
            logo_file = request.files['logo']
            if logo_file.filename != '':
                logo_path = os.path.join(LOGOS_FOLDER, f"{company_name}.png")
                logo_file.save(logo_path)
        
        return jsonify({'success': True, 'message': f'Added company: {company_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/companies/<company_name>', methods=['DELETE'])
def delete_company(company_name):
    """Delete a company and all its data"""
    try:
        # Clear vectorstore cache
        clear_company_vectorstore_cache(company_name)
        
        # Delete PDFs
        company_path = os.path.join(UPLOAD_FOLDER, company_name)
        if os.path.exists(company_path):
            shutil.rmtree(company_path)
        
        # Delete vectorstore
        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
        vectorstore_path = os.path.join(VECTORSTORE_ROOT, company_name)
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
        
        # Delete logo
        logo_path = os.path.join(LOGOS_FOLDER, f"{company_name}.png")
        if os.path.exists(logo_path):
            os.remove(logo_path)
        
        return jsonify({'success': True, 'message': f'Deleted all data for {company_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        pdf_file.save(save_path)
        
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
    """Rebuild knowledge base for company"""
    try:
        from ingest import ingest_company_pdfs
        
        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
        vectorstore_path = os.path.join(VECTORSTORE_ROOT, company_name)
        
        # Clear the cached vectorstore
        clear_company_vectorstore_cache(company_name)
        
        # Remove existing vectorstore
        if os.path.exists(vectorstore_path):
            try:
                shutil.rmtree(vectorstore_path, ignore_errors=True)
                time.sleep(2)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {cleanup_error}")
        
        os.makedirs(vectorstore_path, exist_ok=True)
        
        # Run the ingestion
        vectordb = ingest_company_pdfs(company_name, persist_directory=vectorstore_path)
        
        return jsonify({'success': True, 'message': 'Knowledge base updated successfully!'})
        
    except Exception as e:
        clear_company_vectorstore_cache(company_name)
        error_msg = str(e)
        if "no such table: tenants" in error_msg:
            return jsonify({'error': 'Database corruption detected. Please try again.'}), 500
        else:
            return jsonify({'error': f'Error: {error_msg}'}), 500

@app.route('/api/companies/<company_name>/ask', methods=['POST'])
def ask_company_question(company_name):
    """Ask a question to a specific company"""
    try:
        data = request.get_json()
        query = data.get('question', '')
        
        if not query:
            return jsonify({'error': 'Question is required'}), 400
        
        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
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
        if "no such table: tenants" in error_msg:
            clear_company_vectorstore_cache(company_name)
            return jsonify({'error': 'Database error detected. Please click Relearn PDFs to rebuild the knowledge base.'}), 500
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
        
        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
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
                    if "no such table: tenants" in error_msg:
                        clear_company_vectorstore_cache(company)
                        responses.append({
                            'company': company,
                            'error': 'Database error detected. Please click Relearn PDFs to rebuild the knowledge base.',
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
    return send_from_directory('static', 'index.html')

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

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
