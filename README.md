# AI Market Research Assistant — Prototype

Capstone / Applied AI Mini Project — TEC 406 (Foundations of Generative AI and Agentic AI)

## What this is
A working Streamlit prototype of the multi-agent RAG pipeline described in the
accompanying report (`AI_Market_Research_Assistant_Capstone_Report.docx`):
Orchestrator → Search Agent → Retrieval/Summarization Agent → Analyst Agent
(trends + SWOT) → Evaluation Guardrail → Writer Agent.

## Setup
```bash
pip install -r requirements.txt
streamlit run market_research_assistant_app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## API key
Paste your Anthropic API key into the sidebar, or set it as an environment
variable before launching:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run market_research_assistant_app.py
```

**No key? No problem.** The app runs in demo mode with clearly-labeled
placeholder text so the full pipeline (search → summarize → trends → SWOT →
report) can still be walked through end-to-end for a class demo.

## Notes for the team
- Web search uses the free `duckduckgo-search` package — no search API key
  needed. Swap in SerpAPI/Tavily for more reliable results in a real deployment.
- The "retrieval" step in this prototype summarizes search snippets directly
  rather than running a full embedding + vector-store pipeline (FAISS/Chroma).
  That's the natural next upgrade if you want to demonstrate true RAG over
  longer documents (e.g., full competitor web pages or PDF reports) — the
  report's architecture section describes where that slots in.
- The guardrail in `flag_low_confidence()` is a simple heuristic for the demo;
  the report proposes an LLM-as-judge groundedness check as the production version.
- Each function in the script maps directly to one "agent" in the report's
  architecture and pipeline diagrams — use that mapping when presenting.
