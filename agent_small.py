#!/usr/bin/env python3
"""
Compatibility wrapper for Hugging Face Space.
This file redirects to the new hf_space/hf_agent_small.py implementation.
"""

# Import everything from the new location
from hf_space.hf_agent_small import *

# This maintains backward compatibility with HF Space configurations
# that expect to import from agent_small.py