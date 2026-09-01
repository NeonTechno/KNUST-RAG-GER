#!/usr/bin/env python3
"""
Hugging Face Space entrypoint for KNUST RAG-GAR AI Agent.
This serves as the main application file for the HF Space.
"""

import re
from pathlib import Path
import gradio as gr
import os
from huggingface_hub import InferenceClient

# Set HF_TOKEN from Space secrets if available
if "HF_TOKEN" not in os.environ:
    hf_token = os.environ.get("HF_TOKEN_FROM_SPACE") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

# Load KNUST documents
DATA = Path(__file__).parent / "data"

def load_docs():
    docs = []
    for p in sorted(DATA.glob("*.txt")):
        for part in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")):
            if part.strip():
                docs.append(f"[{p.name}]\n{part.strip()}")
    return docs

DOCS = load_docs()

def knust_knowledge_search(query):
    """Search the local KNUST admissions and campus knowledge base."""
    terms = {x.lower() for x in re.findall(r"[A-Za-z0-9']+", query) if len(x) > 2}
    ranked = []
    for doc in DOCS:
        score = sum(t in doc.lower() for t in terms)
        if score:
            ranked.append((score, doc))
    ranked.sort(reverse=True)
    return "\n\n---\n\n".join(x[1] for x in ranked[:5]) or "No matching local information."

def get_hf_client():
    """Get Hugging Face InferenceClient with proper token handling."""
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    if token:
        return InferenceClient(token=token, model=model, provider="auto")
    else:
        return InferenceClient(model=model, provider="auto")

def chat(message, history):
    """Handle chat messages with KNUST RAG + HF inference."""
    context = knust_knowledge_search(message)
    
    prompt = f"""You are the KNUST E-Learning Centre AI Assistant.
Use the provided KNUST knowledge base context to answer questions accurately.
Do not invent university policies, fees, dates, requirements, or URLs.
If the context doesn't answer the question, explicitly say the information is not available.

KNUST Knowledge Context:
{context}

Question: {message}

Answer:"""
    
    try:
        client = get_hf_client()
        response = client.text_generation(prompt, max_new_tokens=500, temperature=0.2)
        return response.strip()
    except Exception as e:
        if context and context != "No matching local information.":
            return f"Based on available information: {context}"
        else:
            return "The AI assistant is temporarily unavailable. Please try again."

# Define Gradio interface at module level
demo = gr.ChatInterface(
    fn=chat,
    title="KNUST E-Learning Centre AI Assistant",
    description="Ask about admissions, procedures and campus navigation.",
    examples=[
        ["How do I apply for admission to KNUST?"],
        ["What are the admission requirements?"],
        ["What is the application fee?"],
        ["How do I access the E-Learning platform?"],
    ]
)

# For compatibility with HF Spaces that expect 'app' variable
app = demo

if __name__ == "__main__":
    demo.launch()