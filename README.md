# AI Market Research Assistant — Prototype

Capstone / Applied AI Mini Project — TEC 406 (Foundations of Generative AI and Agentic AI)

## What this is
A working Streamlit prototype of the multi-agent RAG pipeline described in the
accompanying report (`AI_Market_Research_Assistant_Capstone_Report.docx`):
Orchestrator → Search Agent → Retrieval/Summarization Agent → Analyst Agent
(trends + SWOT) → Evaluation Guardrail → Writer Agent.

## No API key needed
This runs entirely for free by default:
- **Web search** uses `duckduckgo-search` — no account, no key.
- **The "AI" steps** (summarization, trend extraction, SWOT generation) run on
  **`google/flan-t5-base`**, a small, open-source, instruction-tuned model
  downloaded once from Hugging Face and executed locally on your own machine
  via the `transformers` library. No account, no key, no per-call cost, and
  after the first download it works fully offline.

An Anthropic API key is **optional** — only add one (in the sidebar) if you
want noticeably higher-quality output from Claude instead of the local model.
The rest of the app (search, agent orchestration, report structure) is
identical either way.

## Setup
```bash
pip install -r requirements.txt
streamlit run market_research_assistant_app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

**First run:** downloading the local model (~1 GB) takes a few minutes
depending on your connection; after that it's cached and loads instantly.
If your machine is slow, open `market_research_assistant_app.py` and change
`LOCAL_MODEL_NAME` from `"google/flan-t5-base"` to `"google/flan-t5-small"`
(~300 MB, faster, slightly lower quality) — good enough for a class demo.

## Using Claude instead (optional)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run market_research_assistant_app.py
```
Or just paste the key into the sidebar at runtime. If no key is given, the
app automatically falls back to the free local model — nothing else to configure.

## Notes for the team
- If neither `transformers`/`torch` are installed nor an API key is given,
  the app still runs end-to-end in a clearly-labeled offline demo mode with
  placeholder text — useful for a quick walkthrough without any setup at all.
- The "retrieval" step in this prototype summarizes search snippets directly
  rather than running a full embedding + vector-store pipeline (FAISS/Chroma).
  That's the natural next upgrade if you want to demonstrate true RAG over
  longer documents (e.g., full competitor web pages or PDF reports) — the
  report's architecture section describes where that slots in.
- The guardrail in `flag_low_confidence()` is a simple heuristic for the demo;
  the report proposes an LLM-as-judge groundedness check as the production version.
- Each function in the script maps directly to one "agent" in the report's
  architecture and pipeline diagrams — use that mapping when presenting.
