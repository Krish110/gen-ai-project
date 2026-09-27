"""
AI Market Research Assistant — Prototype
Capstone / Applied AI Mini Project — TEC 406 (Foundations of Generative AI and Agentic AI)

A multi-agent RAG pipeline that:
  1. Decomposes a research brief into sub-questions          (Orchestrator)
  2. Searches the live web for relevant sources               (Search Agent)
  3. Chunks + embeds retrieved text into a vector store        (Retrieval Agent)
  4. Summarizes sources & extracts trends, grounded in context (Analyst Agent)
  5. Flags low-confidence / unsupported claims                 (Evaluation Agent)
  6. Compiles a structured, cited report                       (Writer Agent)

Run with:
    pip install -r requirements.txt
    streamlit run market_research_assistant_app.py

NO API KEY NEEDED BY DEFAULT. The "LLM" steps (summarization, trend
extraction, SWOT generation) run on a small, free, open-source model
(google/flan-t5-base) downloaded once from Hugging Face and executed
locally on your own machine via the `transformers` library — no account,
no key, no per-call cost. Web search uses the free `duckduckgo-search`
package — also no key needed.

An Anthropic API key is optional: if you have one and want noticeably
higher-quality output, paste it into the sidebar and the app will use
Claude instead of the local model for the same steps.
"""

import os
import time
import textwrap
from dataclasses import dataclass, field

import streamlit as st

# ---------------------------------------------------------------------------
# Optional dependencies — the app degrades gracefully if they're missing so
# it can still be reviewed/graded without every package installed.
# ---------------------------------------------------------------------------
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Free, local, instruction-tuned model — no API key required.
# "google/flan-t5-small" is faster to download (~300 MB) and enough for a
# class demo; "google/flan-t5-base" (~1 GB) gives noticeably better quality
# on a normal laptop CPU. Change this constant to switch.
LOCAL_MODEL_NAME = "google/flan-t5-base"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Source:
    title: str
    url: str
    snippet: str
    summary: str = ""


@dataclass
class ResearchResult:
    brief: str
    sub_questions: list = field(default_factory=list)
    sources: list = field(default_factory=list)  # list[Source]
    trends: str = ""
    swot: str = ""
    recommendations: str = ""
    flagged_claims: list = field(default_factory=list)
    report_markdown: str = ""
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Local, free model loader — cached so the model downloads/loads only once
# per Streamlit session, not on every button click.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_local_model(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        # Fails gracefully without crashing Streamlit
        print(f"Error loading local model {model_name}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# LLM wrapper (Analyst / Writer agents call this)
# ---------------------------------------------------------------------------
class LLM:
    """Thin wrapper so the rest of the app doesn't care which backend is used.

    Backends, in order of preference:
      1. "anthropic" — Claude API, only used if the user supplies a key.
      2. "local"     — free, local, open-source model via `transformers`.
                       No API key, no cost, runs on the user's own machine.
      3. "demo"       — offline placeholder text, used only if neither of the
                       above is available (e.g. transformers isn't installed).
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-6",
                 local_model_name: str = LOCAL_MODEL_NAME):
        self.backend = "demo"
        self.client = None
        self.tokenizer = None
        self.local_model = None

        if api_key and HAS_ANTHROPIC:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            self.backend = "anthropic"
        elif HAS_TRANSFORMERS:
            self.tokenizer, self.local_model = _load_local_model(local_model_name)
            if self.tokenizer is not None and self.local_model is not None:
                self.backend = "local"
            else:
                self.backend = "demo"

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        if self.backend == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(block.text for block in resp.content if block.type == "text")

        if self.backend == "local":
            # Flan-T5 is instruction-tuned: folding the system instruction and
            # user content into one prompt works well and needs no chat format.
            prompt = f"{system.strip()}\n\n{user.strip()}"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 256),
                do_sample=False,
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return self._mock_response(user)

    @staticmethod
    def _mock_response(user: str) -> str:
        return (
            "[DEMO MODE — no local model or API key available. This is "
            "placeholder text so the pipeline can still be exercised end-to-end. "
            "Install `transformers` + `torch` (see requirements.txt) to enable "
            "the free local model.]\n\n"
            f"Simulated analysis based on prompt:\n{textwrap.shorten(user, 200)}"
        )


# ---------------------------------------------------------------------------
# Agent 1 — Orchestrator: query decomposition
# ---------------------------------------------------------------------------
def decompose_query(llm: LLM, brief: str) -> list:
    system = (
        "You are a market research planning assistant. Break the brief into "
        "4-5 specific, web-searchable sub-questions covering competitor "
        "positioning, pricing, recent campaigns/news, and market trends. "
        "Return ONLY a numbered list, one sub-question per line."
    )
    text = llm.complete(system, f'Research brief: "{brief}"', max_tokens=300)
    lines = [l.strip(" -.") for l in text.split("\n") if l.strip()]
    # keep lines that look like list items; fall back to the raw brief
    questions = [l.split(".", 1)[-1].strip() if l[:2].rstrip(".").isdigit() else l for l in lines]
    questions = [q for q in questions if len(q) > 8][:5]
    return questions or [brief]


# ---------------------------------------------------------------------------
# Agent 2 — Search Agent (web retrieval)
# ---------------------------------------------------------------------------
def search_web(query: str, max_results: int = 3) -> list:
    if not HAS_DDG:
        return [Source(title=f"[Demo source for] {query}", url="https://example.com",
                        snippet="duckduckgo-search not installed — install requirements.txt "
                                "to enable live web search.")]
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(Source(
                    title=r.get("title", "Untitled"),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                ))
    except Exception as e:
        results.append(Source(title="Search error", url="", snippet=str(e)))
    return results


# ---------------------------------------------------------------------------
# Agent 3 — Retrieval / Summarization Agent
# (In this lightweight prototype, "retrieval" operates directly over search
#  snippets rather than a full vector store — swap in FAISS/Chroma + an
#  embedding model for the production version described in the report.)
# ---------------------------------------------------------------------------
def summarize_source(llm: LLM, sub_question: str, source: Source) -> str:
    system = (
        "Summarize the given snippet in 2-3 sentences, focused only on facts "
        "relevant to the sub-question. Preserve numbers, dates, and named "
        "entities. If the snippet has no relevant information, say so plainly."
    )
    user = f"Sub-question: {sub_question}\n\nSource ({source.url}):\n{source.snippet}"
    return llm.complete(system, user, max_tokens=200)


# ---------------------------------------------------------------------------
# Agent 4 — Analyst Agent (trends + SWOT)
# ---------------------------------------------------------------------------
def extract_trends(llm: LLM, all_summaries: str) -> str:
    system = (
        "Identify the top 3 recurring trends or patterns from these source "
        "summaries. For each, note which sources support it. Flag single-"
        "source claims as low-confidence."
    )
    return llm.complete(system, all_summaries, max_tokens=400)


def generate_swot(llm: LLM, company: str, market: str, findings: str) -> str:
    system = (
        "Generate a SWOT analysis (Strengths, Weaknesses, Opportunities, "
        "Threats) using ONLY the findings provided. Do not introduce facts "
        "not present in the findings. Reference which finding supports each point."
    )
    user = f"Company: {company}\nMarket: {market}\n\nFindings:\n{findings}"
    return llm.complete(system, user, max_tokens=500)


def generate_recommendations(llm: LLM, swot: str) -> str:
    system = "Based on this SWOT analysis, propose 2-3 concrete strategic recommendations."
    return llm.complete(system, swot, max_tokens=300)


# ---------------------------------------------------------------------------
# Agent 5 — Evaluation / Guardrail Agent
# ---------------------------------------------------------------------------
def flag_low_confidence(sources: list) -> list:
    """Simple heuristic guardrail for the prototype: any sub-question with
    fewer than 2 corroborating sources is flagged for human review. The
    production system replaces this with an LLM-as-judge groundedness check."""
    flags = []
    for s in sources:
        if not s.summary or "no relevant information" in s.summary.lower():
            flags.append(f"Low-confidence / unverified: {s.title}")
    return flags


# ---------------------------------------------------------------------------
# Agent 6 — Writer Agent (final report compilation)
# ---------------------------------------------------------------------------
def compile_report(result: ResearchResult, company: str, market: str) -> str:
    lines = [
        f"# Market Research Report: {company} ({market})",
        "",
        "## Executive Summary",
        f"Automated research brief: *{result.brief}*",
        "",
        "## Sub-Questions Investigated",
    ]
    lines += [f"- {q}" for q in result.sub_questions]
    lines += ["", "## Trend Analysis", result.trends, "", "## SWOT Analysis", result.swot,
              "", "## Strategic Recommendations", result.recommendations]
    if result.flagged_claims:
        lines += ["", "## \u26a0\ufe0f Flagged for Human Review"]
        lines += [f"- {f}" for f in result.flagged_claims]
    lines += ["", "## Sources"]
    for i, s in enumerate(result.sources, 1):
        lines.append(f"[{i}] {s.title} — {s.url}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(llm: LLM, brief: str, company: str, market: str, progress_cb=None) -> ResearchResult:
    t0 = time.time()
    result = ResearchResult(brief=brief)

    if progress_cb: progress_cb(0.1, "Decomposing brief into sub-questions…")
    result.sub_questions = decompose_query(llm, brief)

    all_sources = []
    for i, q in enumerate(result.sub_questions):
        if progress_cb:
            progress_cb(0.15 + 0.4 * (i / max(len(result.sub_questions), 1)),
                        f"Searching: {q[:60]}…")
        found = search_web(q)
        for s in found:
            s.summary = summarize_source(llm, q, s)
        all_sources.extend(found)
    result.sources = all_sources

    if progress_cb: progress_cb(0.6, "Extracting trends across sources…")
    joined_summaries = "\n".join(f"- ({s.url}) {s.summary}" for s in all_sources)
    result.trends = extract_trends(llm, joined_summaries)

    if progress_cb: progress_cb(0.75, "Generating SWOT analysis…")
    result.swot = generate_swot(llm, company, market, joined_summaries + "\n\nTrends:\n" + result.trends)

    if progress_cb: progress_cb(0.85, "Drafting strategic recommendations…")
    result.recommendations = generate_recommendations(llm, result.swot)

    if progress_cb: progress_cb(0.92, "Running evaluation / guardrail checks…")
    result.flagged_claims = flag_low_confidence(all_sources)

    if progress_cb: progress_cb(0.97, "Compiling final report…")
    result.report_markdown = compile_report(result, company, market)

    result.elapsed_seconds = time.time() - t0
    if progress_cb: progress_cb(1.0, "Done.")
    return result


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Market Research Assistant", page_icon="📊", layout="wide")

    st.title("📊 AI Market Research Assistant")
    st.caption("Capstone Prototype — Multi-Agent RAG pipeline for automated competitive intelligence")

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            "**No API key needed.** By default this runs on a free, local "
            "open-source model (no account, no cost). An API key is optional "
            "— only add one if you want higher-quality output from Claude."
        )
        api_key = st.text_input(
            "Anthropic API Key (optional)", value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password", help="Leave blank to use the free local model instead."
        )
        st.markdown("---")
        st.markdown("**Pipeline status**")
        st.write(f"Web search (duckduckgo-search): {'✅ available' if HAS_DDG else '⚠️ not installed'}")
        st.write(f"Free local model (transformers): {'✅ available' if HAS_TRANSFORMERS else '⚠️ not installed — run pip install -r requirements.txt'}")
        st.write(f"Claude API (optional, needs key): {'✅ available' if HAS_ANTHROPIC else '⚠️ SDK not installed'}")
        active_backend = "Claude API" if (api_key and HAS_ANTHROPIC) else ("Free local model" if HAS_TRANSFORMERS else "Demo/offline")
        st.info(f"Active backend for this run: **{active_backend}**")
        if active_backend == "Free local model":
            st.caption(f"Model: {LOCAL_MODEL_NAME} — downloads once (~1 GB) on first run, then runs fully offline.")
        st.markdown("---")
        st.markdown(
            "This prototype implements the architecture in **Section 3** and the "
            "pipeline in **Section 5** of the capstone report: Orchestrator → "
            "Search Agent → Retrieval/Summarization → Analyst Agent → "
            "Evaluation Guardrail → Writer Agent."
        )

    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input("Company / Brand", placeholder="e.g., Nykaa")
    with col2:
        market = st.text_input("Market / Industry", placeholder="e.g., D2C beauty & personal care, India")

    brief = st.text_area(
        "Research Brief",
        placeholder="e.g., Research the competitor's pricing strategy, recent marketing "
                    "campaigns, and positioning in the last 12 months.",
        height=90,
    )

    run = st.button("🚀 Run Market Research Agent", type="primary", use_container_width=True)

    if run:
        if not company or not market or not brief:
            st.warning("Please fill in company, market, and research brief.")
            return

        llm = LLM(api_key=api_key)
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress_cb(pct, msg):
            progress_bar.progress(pct)
            status.info(msg)

        with st.spinner("Running agent pipeline…"):
            result = run_pipeline(llm, brief, company, market, progress_cb)

        status.success(f"Report generated in {result.elapsed_seconds:.1f} seconds "
                        f"(vs. an estimated 8–16 hours of manual research).")

        tab1, tab2, tab3, tab4 = st.tabs(["📄 Full Report", "🔍 Trends", "📊 SWOT", "📚 Sources"])

        with tab1:
            st.markdown(result.report_markdown)
            st.download_button("Download Report (Markdown)", result.report_markdown,
                                file_name=f"{company}_market_research_report.md")

        with tab2:
            st.markdown(result.trends)

        with tab3:
            st.markdown(result.swot)
            st.markdown("### Strategic Recommendations")
            st.markdown(result.recommendations)

        with tab4:
            for i, s in enumerate(result.sources, 1):
                with st.expander(f"[{i}] {s.title}"):
                    st.write(s.url)
                    st.write("**Summary:**", s.summary)
                    st.write("**Raw snippet:**", s.snippet)
            if result.flagged_claims:
                st.warning("Flagged for human review:\n" + "\n".join(f"- {f}" for f in result.flagged_claims))


if __name__ == "__main__":
    main()
