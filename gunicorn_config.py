"""
Gunicorn Configuration for MOM-Bot Production Deployment
"""

import multiprocessing
import os

# Server Socket
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
backlog = 2048

# Worker Processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'sync'  # Use 'sync' for CPU-bound ML tasks
worker_connections = 1000
max_requests = 1000  # Restart workers after this many requests (prevents memory leaks)
max_requests_jitter = 50  # Add randomness to avoid all workers restarting simultaneously
timeout = 300  # 5 minutes - ML processing can take time
keepalive = 5

# Process Naming
proc_name = 'mom-bot'

# Logging
accesslog = os.getenv('GUNICORN_ACCESS_LOG', '-')  # '-' means stdout
errorlog = os.getenv('GUNICORN_ERROR_LOG', '-')    # '-' means stderr
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Server Mechanics
daemon = False  # Run in foreground for better process management
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed for HTTPS)
keyfile = os.getenv('SSL_KEY_FILE', None)
certfile = os.getenv('SSL_CERT_FILE', None)

# Server Hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting MOM-Bot Gunicorn server")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading MOM-Bot workers")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("MOM-Bot server is ready. Listening on: %s", bind)

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing.")

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info("Worker received INT or QUIT signal")

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT signal")
