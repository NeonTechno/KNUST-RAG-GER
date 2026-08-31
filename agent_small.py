import re
from pathlib import Path
import gradio as gr
from smolagents import CodeAgent, InferenceClientModel, Tool, DuckDuckGoSearchTool

DATA = Path(__file__).parent / "data"

def load_docs():
    docs = []
    for p in sorted(DATA.glob("*.txt")):
        for part in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")):
            if part.strip():
                docs.append(f"[{p.name}]\n{part.strip()}")
    return docs

DOCS = load_docs()

class KNUSTSearch(Tool):
    name = "knust_knowledge_search"
    description = "Search the local KNUST admissions and campus knowledge base."
    inputs = {"query": {"type": "string", "description": "KNUST question or keywords"}}
    output_type = "string"

    def forward(self, query):
        terms = {x.lower() for x in re.findall(r"[A-Za-z0-9']+", query) if len(x) > 2}
        ranked = []
        for doc in DOCS:
            score = sum(t in doc.lower() for t in terms)
            if score:
                ranked.append((score, doc))
        ranked.sort(reverse=True)
        return "\n\n---\n\n".join(x[1] for x in ranked[:5]) or "No matching local information."

model = InferenceClientModel(model_id="Qwen/Qwen2.5-7B-Instruct")
agent = CodeAgent(model=model, tools=[KNUSTSearch(), DuckDuckGoSearchTool()], max_steps=6)

def chat(message, history):
    prompt = f"""You are the KNUST E-Learning Centre AI Assistant.
Use knust_knowledge_search first for KNUST questions. Use web search for current public facts.
Do not invent university policies, fees, dates or requirements. If evidence is insufficient, say so.

Question: {message}"""
    try:
        return agent.run(prompt)
    except Exception:
        return "The AI assistant is temporarily unavailable. Please try again."

demo = gr.ChatInterface(
    fn=chat,
    title="KNUST E-Learning Centre AI Assistant",
    description="Ask about admissions, procedures and campus navigation.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
