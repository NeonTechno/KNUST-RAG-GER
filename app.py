#!/usr/bin/env python3
"""
Vercel entrypoint for KNUST RAG-GAR API.
This file serves as the main application entrypoint for Vercel deployments.
"""

# Import the FastAPI app from api/index.py
from api.index import app

# This makes the app available for Vercel's automatic detection
# Vercel expects to find 'app' in the root app.py file
