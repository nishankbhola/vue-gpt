# backend/gunicorn.conf.py
bind = "127.0.0.1:8000"  # ✅ This keeps it private, behind Nginx
workers = 2  # Reduced for shared hosting
worker_class = "sync"
timeout = 300
client_max_body_size = "500m" 
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
user = "www-data"  # Add this for proper permissions
group = "www-data"
