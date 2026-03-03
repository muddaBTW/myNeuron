"""
Vercel Serverless Entry Point
Routes all API requests to the FastAPI application.
"""

import sys
import os

# Add the backend directory to the Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app
