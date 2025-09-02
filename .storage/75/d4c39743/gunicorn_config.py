import os

# Gunicorn configuration for production deployment
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
workers = int(os.environ.get('WEB_CONCURRENCY', 2))
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
preload_app = True