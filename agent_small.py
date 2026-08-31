#!/usr/bin/env python3
"""
KNUST FAQ Agent using SmolAgents, simple keyword lookup, and Gradio UI.
"""

import os
import gradio as gr
from smolagents import ToolCallingAgent, Tool, DuckDuckGoSearchTool, WikipediaSearchTool

# -------------------------
# 1. Load KNUST FAQ into memory
# -------------------------
FAQ_PATH = "knust_faq.txt"

def load_faq():
    if not os.path.exists(FAQ_PATH):
        # Create a dummy FAQ if file missing
        with open(FAQ_PATH, "w", encoding="utf-8") as f:
            f.write("""KNUST FAQ
What is KNUST? Kwame Nkrumah University of Science and Technology.
Where is KNUST located? Kumasi, Ghana.
How to apply for admission? Visit the admissions portal.
""")
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    # Simple chunking by paragraphs
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks

faq_chunks = load_faq()

# -------------------------
# 2. Define custom tools
# -------------------------
class DocumentLookupTool(Tool):
    name = "document_lookup"
    description = "Lookup information in the KNUST FAQ store using simple keyword match."
    inputs = {"question": {"type": "string", "description": "The question to ask."}}
    output_type = "string"

    def forward(self, question: str) -> str:
        # Simple case-insensitive substring match
        matches = [chunk for chunk in faq_chunks if question.lower() in chunk.lower()]
        if not matches:
            return "No relevant information found in the FAQ."
        return "\n\n---\n\n".join(matches[:3])  # return top 3 matches

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
        hist_text = "\n".join([f"User: {u}\\nAssistant: {a}" for u, a in history])
        prompt = f"{hist_text}\\nUser: {message}"
    response = agent.run(prompt)
    return response

demo = gr.ChatInterface(
    fn=chat_fn,
    title="KNUST FAQ Agent",
    description="Ask questions about KNUST or general knowledge.",
    theme="soft",
)

app = demo

if __name__ == "__main__":
    app.launch()