#!/usr/bin/env python3
"""
Hugging Face Space entrypoint for KNUST RAG-GAR AI Agent.
This serves as the main application file for the HF Space.
"""

import os
# Set HF_TOKEN from Space secrets if available
if "HF_TOKEN" not in os.environ:
    # Try to get from Space environment variables
    hf_token = os.environ.get("HF_TOKEN_FROM_SPACE") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

# Import the demo from hf_agent_small
from hf_agent_small import demo

# For compatibility with HF Spaces that expect 'app' variable
app = demo

if __name__ == "__main__":
    demo.launch()