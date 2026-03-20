"""
Application entry point.

    python main.py
    uvicorn main:app --reload --port 8000
"""

import uvicorn
from src.drivers.config import API_HOST, API_PORT, API_RELOAD
from src.drivers.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
