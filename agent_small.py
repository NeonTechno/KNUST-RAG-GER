#!/usr/bin/env python3
"""
Compatibility wrapper for Hugging Face Space.
This file redirects to the main app.py implementation.
"""

# Import from app.py for backward compatibility
from app import app, demo

# This maintains backward compatibility with configurations
# that expect to import from agent_small.py