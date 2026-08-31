#!/usr/bin/env python3
"""KNUST E-Learning Centre AI Assistant — Hugging Face Space entrypoint."""

from pathlib import Path
import re
import gradio as gr
from smolagents import CodeAgent, InferenceClientModel, Tool, DuckDuckGoSearchTool

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def load_knowledge():
    chunks = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for chunk in re.split(r"\n\s*\n", text):
            chunk = chunk.strip()
            if chunk:
                chunks.append(f"[{path.name}]\n{chunk}")
    return chunks

KNOWLEDGE = load_knowledge()

class KNUSTRetrievalTool(Tool):
    name = "knust_knowledge_search"
    description = (
        "Search the local KNUST E-Learning Centre knowledge base for admissions, "
        "fees, requirements, procedures, navigation and campus information."
    )
    inputs = {"query": {"type": "string", "description": "KNUST-related question or keywords."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        terms = {t.lower() for t in re.findall(r"[A-Za-z0-9']+", query) if len(t) > 2}
        scored = []
        for chunk in KNOWLEDGE:
            haystack = chunk.lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return "No directly relevant local KNUST information was found."
        return "\n\n---\n\n".join(chunk for _, chunk in scored[:5])

def build_agent():
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-7B-Instruct")
    return CodeAgent(
        model=model,
        tools=[KNUSTRetrievalTool(), DuckDuckGoSearchTool()],
        max_steps=6,
    )

agent = build_agent()

def chat_fn(message, history):
    history = history or []
    previous = []
    for item in history[-8:]:
        if isinstance(item, dict):
            role, content = item.get("role", ""), item.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str):
                previous.append(f"{role.title()}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            previous.append(f"User: {item[0]}\nAssistant: {item[1]}")
    prompt = f"""You are the KNUST E-Learning Centre AI Assistant.
Use knust_knowledge_search first for KNUST-specific questions.
Use web search only when current public information is needed.
Never invent admission dates, fees, requirements, links or university policies.
If the knowledge base is insufficient, say so and direct the user to an official KNUST source.

Conversation:
{"\n".join(previous)}

User:
{message}
"""
    try:
        return agent.run(prompt)
    except Exception as exc:
        return f"I’m temporarily unable to process that request. Please try again. ({type(exc).__name__})"

demo = gr.ChatInterface(
    fn=chat_fn,
    title="KNUST E-Learning Centre AI Assistant",
    description="Admissions, procedures and campus navigation assistant.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
