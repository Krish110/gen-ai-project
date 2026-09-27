"""
AI Market Research Assistant — Prototype
"""

import os
import time
import textwrap
from dataclasses import dataclass, field
import streamlit as st

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

try:
    from huggingface_hub import InferenceClient
    HAS_HF = True
except ImportError:
    HAS_HF = False

# A highly capable open-source model available on HF's free inference tier
HF_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

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
    sources: list = field(default_factory=list)
    trends: str = ""
    swot: str = ""
    recommendations: str = ""
    flagged_claims: list = field(default_factory=list)
    report_markdown: str = ""
    elapsed_seconds: float = 0.0

class LLM:
    def __init__(self, hf_token: str = "", model: str = HF_MODEL_NAME):
        self.backend = "demo"
        self.client = None

        if hf_token and HAS_HF:
            self.client = InferenceClient(model=model, token=hf_token)
            self.backend = "huggingface"

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        if self.backend == "huggingface":
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                import streamlit as st
                st.error(f"Hugging Face API Error: {str(e)}")
                return f"Error: {str(e)}"

        return self._mock_response(user)

    @staticmethod
    def _mock_response(user: str) -> str:
        return (
            "[DEMO MODE — No API keys provided.]\n\n"
            f"Simulated analysis based on prompt:\n{textwrap.shorten(user, 200)}"
        )


def decompose_query(llm: LLM, brief: str, company: str, market: str) -> list:
    system = (
        "You are a research planner. Break the brief into 4 specific, highly targeted search engine queries. "
        "Return ONLY a numbered list of keywords. DO NOT use full sentences or conversational questions. "
        "You MUST include the company name in every query."
    )
    user_prompt = f"Company: {company}\nMarket: {market}\nBrief: {brief}\n\nGenerate 4 keyword-based search queries:"
    text = llm.complete(system, user_prompt, max_tokens=300)
    lines = [l.strip(" -.*") for l in text.split("\n") if l.strip()]
    questions = [l.split(".", 1)[-1].strip() if l[:2].rstrip(".").isdigit() else l for l in lines]
    return [q for q in questions if len(q) > 8][:5] or [brief]

def search_web(query: str, max_results: int = 3) -> list:
    if not HAS_DDG:
        return [Source(title="Demo", url="https://example.com", snippet="Install duckduckgo-search")]
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(Source(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", "")))
    except Exception as e:
        results.append(Source(title="Search error", url="", snippet=str(e)))
    return results

def summarize_source(llm: LLM, sub_question: str, source: Source) -> str:
    system = "Summarize the snippet in 2 sentences focused on facts relevant to the search query. If irrelevant, say so plainly."
    return llm.complete(system, f"Search Query: {sub_question}\nSnippet: {source.snippet}", max_tokens=150)

def extract_trends(llm: LLM, all_summaries: str) -> str:
    system = "Identify the top 3 recurring trends from these summaries. Keep it concise."
    return llm.complete(system, all_summaries, max_tokens=300)

def generate_swot(llm: LLM, company: str, market: str, findings: str) -> str:
    system = "Generate a brief SWOT analysis using ONLY the provided findings. Do not invent facts."
    return llm.complete(system, f"Company: {company}\nMarket: {market}\nFindings:\n{findings}", max_tokens=400)

def generate_recommendations(llm: LLM, swot: str) -> str:
    system = "Based on this SWOT, propose 2 concrete strategic recommendations."
    return llm.complete(system, swot, max_tokens=250)

def flag_low_confidence(sources: list) -> list:
    return [f"Low-confidence: {s.title}" for s in sources if not s.summary or "irrelevant" in s.summary.lower()]

def compile_report(result: ResearchResult, company: str, market: str) -> str:
    lines = [f"# Market Research: {company} ({market})", "", "## Search Queries Used"]
    lines += [f"- {q}" for q in result.sub_questions]
    lines += ["", "## Trend Analysis", result.trends, "", "## SWOT Analysis", result.swot, "", "## Recommendations", result.recommendations, "", "## Sources"]
    for i, s in enumerate(result.sources, 1):
        lines.append(f"[{i}] {s.url}")
    return "\n".join(lines)

def run_pipeline(llm: LLM, brief: str, company: str, market: str, progress_cb=None) -> ResearchResult:
    t0 = time.time()
    result = ResearchResult(brief=brief)

    if progress_cb: progress_cb(0.1, "Generating search queries…")
    result.sub_questions = decompose_query(llm, brief, company, market)

    all_sources = []
    for i, q in enumerate(result.sub_questions):
        if progress_cb: progress_cb(0.15 + 0.4 * (i / max(len(result.sub_questions), 1)), f"Searching: {q[:60]}…")
        found = search_web(q)
        for s in found: s.summary = summarize_source(llm, q, s)
        all_sources.extend(found)
    result.sources = all_sources

    if progress_cb: progress_cb(0.6, "Extracting trends…")
    joined_summaries = "\n".join(f"- {s.summary}" for s in all_sources)
    result.trends = extract_trends(llm, joined_summaries)

    if progress_cb: progress_cb(0.75, "Generating SWOT…")
    result.swot = generate_swot(llm, company, market, joined_summaries + "\n\n" + result.trends)

    if progress_cb: progress_cb(0.85, "Drafting recommendations…")
    result.recommendations = generate_recommendations(llm, result.swot)

    result.flagged_claims = flag_low_confidence(all_sources)
    result.report_markdown = compile_report(result, company, market)
    result.elapsed_seconds = time.time() - t0
    if progress_cb: progress_cb(1.0, "Done.")
    return result

def main():
    st.set_page_config(page_title="AI Market Research Assistant", page_icon="📊", layout="wide")
    st.title("📊 AI Market Research Assistant")

    with st.sidebar:
        st.header("Configuration")
        hf_token = st.text_input("Hugging Face Token (Free)", type="password", help="Get a free 'Fine-grained' token at huggingface.co/settings/tokens")
        st.info(f"Active backend: **{'Hugging Face API' if hf_token and HAS_HF else 'Demo/Offline'}**")

    col1, col2 = st.columns(2)
    with col1: company = st.text_input("Company / Brand")
    with col2: market = st.text_input("Market / Industry")
    brief = st.text_area("Research Brief", height=90)

    if st.button("🚀 Run Market Research Agent", type="primary", use_container_width=True):
        if not company or not market or not brief:
            st.warning("Please fill in all fields.")
            return

        llm = LLM(hf_token=hf_token)
        progress_bar, status = st.progress(0.0), st.empty()
        
        with st.spinner("Running agent pipeline…"):
            result = run_pipeline(llm, brief, company, market, lambda pct, msg: (progress_bar.progress(pct), status.info(msg)))

        status.success(f"Generated in {result.elapsed_seconds:.1f} seconds.")
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Report", "🔍 Trends", "📊 SWOT", "📚 Sources"])
        with tab1: st.markdown(result.report_markdown)
        with tab2: st.markdown(result.trends)
        with tab3: st.markdown(result.swot); st.markdown("### Recommendations"); st.markdown(result.recommendations)
        with tab4:
            for i, s in enumerate(result.sources, 1):
                with st.expander(f"[{i}] {s.title}"): st.write(s.url); st.write(s.summary)

if __name__ == "__main__":
    main()
