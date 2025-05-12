from binfut import app, background_task
import threading

# Start the background task when the WSGI app is loaded
bg_thread = threading.Thread(target=background_task, daemon=True)
bg_thread.start()

# This is the WSGI entry point that Gunicorn will use
application = app