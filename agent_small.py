#!/usr/bin/env python3
"""
KNUST FAQ Agent using SmolAgents, RAG with ChromaDB, and Gradio UI.
Using a small model for quick testing.
"""

import os
import gradio as gr
from smolagents import ToolCallingAgent, Tool, DuckDuckGoSearchTool, WikipediaSearchTool
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# -------------------------
# 1. Ingest KNUST FAQ into ChromaDB
# -------------------------
def ingest_faq(faq_path: str = "knust_faq.txt"):
    """Load FAQ text, split into chunks, embed, and store in ChromaDB."""
    if not os.path.exists(faq_path):
        # Create a dummy FAQ if file missing
        with open(faq_path, "w", encoding="utf-8") as f:
            f.write("""KNUST FAQ
What is KNUST? Kwame Nkrumah University of Science and Technology.
Where is KNUST located? Kumasi, Ghana.
How to apply for admission? Visit the admissions portal.
""")
    with open(faq_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Simple chunking by paragraphs
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    # Embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.Client()
    try:
        client.delete_collection("knust_faq")
    except Exception:
        pass
    collection = client.create_collection(
        name="knust_faq",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    # Add chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return collection

# Initialize collection globally
collection = ingest_faq()

# -------------------------
# 2. Define custom tools
# -------------------------
class DocumentLookupTool(Tool):
    name = "document_lookup"
    description = "Lookup information in the KNUST FAQ store."
    inputs = {"question": {"type": "string", "description": "The question to ask."}}
    output_type = "string"

    def forward(self, question: str) -> str:
        results = collection.query(query_texts=[question], n_results=3)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant information found in the FAQ."
        return "\n\n---\n\n".join(docs)

class CalculatorTool(Tool):
    name = "calculator"
    description = "A simple calculator for arithmetic expressions."
    inputs = {"expression": {"type": "string", "description": "Math expression to evaluate."}}
    output_type = "string"

    def forward(self, expression: str) -> str:
        try:
            # Safe eval: only allow numbers and basic operators
            allowed = set("0123456789+-*/(). ")
            if not set(expression) <= allowed:
                return "Invalid expression."
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

# -------------------------
# 3. Initialize Agent with tools
# -------------------------
# Using a small model for quick testing
agent = ToolCallingAgent(
    model="sshle/tiny-distilbert-base-uncased-2",  # Small model from HF
    tools=[
        DuckDuckGoSearchTool(),
        WikipediaSearchTool(),
        DocumentLookupTool(),
        CalculatorTool(),
    ],
    max_steps=5,
)

# -------------------------
# 4. Gradio UI
# -------------------------
def chat_fn(message, history):
    # Agent expects a prompt; we can pass conversation as a single string
    prompt = message
    # Optionally include history
    if history:
        hist_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])
        prompt = f"{hist_text}\nUser: {message}"
    response = agent.run(prompt)
    return response

demo = gr.ChatInterface(
    fn=chat_fn,
    title="KNUST FAQ Agent",
    description="Ask questions about KNUST or general knowledge.",
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()