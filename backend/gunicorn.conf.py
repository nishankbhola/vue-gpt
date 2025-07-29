# backend/gunicorn.conf.py
bind = "0.0.0.0:8000"  # Changed from 127.0.0.1 to 0.0.0.0
workers = 2  # Reduced for shared hosting
worker_class = "sync"
timeout = 120
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
user = "www-data"  # Add this for proper permissions
group = "www-data"
