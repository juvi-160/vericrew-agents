import streamlit as st
from ddgs import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg
from fpdf import FPDF
import textwrap
import requests
import re
from collections import Counter
import matplotlib.pyplot as plt

import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

#os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")

if "GROQ_API_KEY" not in os.environ:
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass   

st.set_page_config(page_title="VeriCrew", page_icon=":material/travel_explore:", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink: #16324F;
    --ink-light: #2C4A6E;
    --sage: #4C7A6C;
    --sage-light: #E4EDE9;
    --amber: #A8762C;
    --amber-light: #F5EBDA;
    --parchment: #FAF7F1;
    --paper: #FFFFFF;
    --text-body: #2B2B26;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-body);
}

h1, h2, h3 {
    font-family: 'Lora', serif !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--parchment);
}

[data-testid="stSidebar"] {
    background-color: var(--ink);
}
[data-testid="stSidebar"] * {
    color: #F2EFE7 !important;
}

.vc-logo-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.2rem 0 1.1rem 0;
}
.vc-logo-mark {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background-color: var(--sage);
    color: #FAF7F1 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Lora', serif;
    font-weight: 600;
    font-size: 15px;
    flex-shrink: 0;
}
.vc-logo-text-main {
    font-family: 'Lora', serif;
    font-size: 17px;
    font-weight: 600;
    line-height: 1.1;
}
.vc-logo-text-sub {
    font-size: 11px;
    color: #A9B7C4 !important;
    letter-spacing: 0.3px;
}

.vc-nav-label {
    font-size: 11px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #8FA0AF !important;
    font-weight: 600;
    margin: 0.4rem 0 0.5rem 0;
}

[data-testid="stSidebar"] .stButton>button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    padding: 0.5rem 0.8rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    margin-bottom: 2px;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background-color: rgba(255, 255, 255, 0.07) !important;
}
[data-testid="stSidebar"] button[kind="primary"] {
    background-color: rgba(76, 122, 108, 0.35) !important;
    border-left: 3px solid var(--sage) !important;
    border-radius: 4px !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover {
    background-color: rgba(76, 122, 108, 0.45) !important;
}

.vc-sidebar-footer {
    border-top: 1px solid rgba(255, 255, 255, 0.12);
    padding-top: 0.8rem;
    margin-top: 0.6rem;
}
.vc-status-row {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 12.5px;
    color: #C9D3DA !important;
    margin-bottom: 0.35rem;
}
.vc-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--sage);
    flex-shrink: 0;
}

[data-testid="stAppViewContainer"] .stButton>button {
    background-color: var(--ink) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px;
}
[data-testid="stAppViewContainer"] .stButton>button:hover {
    background-color: var(--ink-light) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--paper);
    border-radius: 10px !important;
    border: 1px solid #E4DFD2 !important;
    box-shadow: 0 1px 3px rgba(22, 50, 79, 0.06);
}

.vc-hero {
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid #E4DFD2;
    margin-bottom: 1.5rem;
}
.vc-hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 12.5px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--sage);
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.vc-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-bottom: 0.6rem;
}
.vc-badge-research {
    background-color: var(--sage-light);
    color: var(--sage);
    border: 1px solid #C6DAD2;
}
.vc-badge-writer {
    background-color: #E9EEF4;
    color: var(--ink);
    border: 1px solid #C9D6E3;
}
.vc-badge-verified {
    background-color: var(--amber-light);
    color: var(--amber);
    border: 1px dashed #C9A45F;
}

.vc-stamp {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--sage);
    border: 1.5px solid var(--sage);
    border-radius: 4px;
    padding: 2px 8px;
    margin-left: 8px;
    transform: rotate(-2deg);
}

.vc-source-link a {
    color: var(--ink) !important;
    text-decoration: none;
    border-bottom: 1px solid #C9D6E3;
}
.vc-source-link a:hover {
    border-bottom-color: var(--ink);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STOPWORDS = set("""a an the and or but is are was were be been being to of in on for with as
by at from this that these those it its it's their his her they he she we you i our your
has have had not no can could will would should may might about into over under more most
also which what when where who whom how than then so such than data ai healthcare""".split())


def fetch_wikipedia_image(topic):
    """Fetches a real thumbnail image for the topic from Wikipedia's free public API. Returns None if not found."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(topic)}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            thumbnail = data.get("thumbnail", {}).get("source")
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
            return thumbnail, page_url
    except Exception:
        pass
    return None, None


def plot_keyword_frequency(text, top_n=10):
    """Builds a simple bar chart of the most frequent meaningful words in the research text."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    if not words:
        return None
    counts = Counter(words).most_common(top_n)
    if not counts:
        return None
    labels, values = zip(*counts)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(labels[::-1], values[::-1], color="#4C7A6C")
    ax.set_xlabel("Mentions")
    ax.set_title("Most frequent keywords in research findings")
    fig.patch.set_facecolor("#FAF7F1")
    ax.set_facecolor("#FAF7F1")
    fig.tight_layout()
    return fig


def add_wrapped_text(pdf, text, width_chars=95, line_height=6):
    """Writes text to the PDF by manually wrapping it with textwrap first,
    avoiding FPDF's internal line-break logic which can crash on certain content."""
    if not text or not text.strip():
        text = "(No content returned by agent)"
    text = text.encode("latin-1", "replace").decode("latin-1")
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            pdf.ln(line_height / 2)
            continue
        wrapped_lines = textwrap.wrap(paragraph, width=width_chars) or [""]
        for line in wrapped_lines:
            pdf.cell(0, line_height, line, new_x="LMARGIN", new_y="NEXT")


def generate_pdf(topic, research_text, report_text, sources=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    add_wrapped_text(pdf, f"VeriCrew Report: {topic}", width_chars=70, line_height=9)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    add_wrapped_text(pdf, "Research findings (Researcher agent)", width_chars=90, line_height=7)
    pdf.set_font("Helvetica", "", 10)
    add_wrapped_text(pdf, research_text)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    add_wrapped_text(pdf, "Verified summary report", width_chars=90, line_height=7)
    pdf.set_font("Helvetica", "", 10)
    add_wrapped_text(pdf, report_text)

    if sources:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        add_wrapped_text(pdf, "Sources", width_chars=90, line_height=7)
        pdf.set_font("Helvetica", "", 10)
        seen = set()
        for src in sources:
            if src["url"] not in seen:
                seen.add(src["url"])
                add_wrapped_text(pdf, f"{src['title']}: {src['url']}")

    return bytes(pdf.output())


last_sources = []


@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Searches the web for a given query/topic and returns a summary of top results.
    Use this to research any topic: news, facts, concepts, current events, and more."""
    global last_sources
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return f"No results found for '{query}'."
        formatted = f"Search results for '{query}':\n\n"
        sources_this_call = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'No title')
            body = r.get('body', 'No description')
            href = r.get('href', '')
            formatted += f"{i}. {title}\n{body}\nSource: {href}\n\n"
            if href:
                sources_this_call.append({"title": title, "url": href})
        last_sources.extend(sources_this_call)
        return formatted
    except Exception as e:
        return f"Search failed: {str(e)}"


@st.cache_resource
def get_crew(word_count=225, research_notes="", writing_notes=""):
    llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    max_tokens=max(300, int(word_count * 1.8))
    )

    researcher = Agent(
        role="Research Specialist",
        goal="Gather accurate, relevant information about the given topic: {topic}",
        backstory="You are a meticulous researcher who searches the web and pulls together "
                  "the most relevant facts and information on any topic, clearly and objectively.",
        tools=[web_search_tool],
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Content Writer",
        goal="Turn research findings into a clear, well organized summary report",
        backstory="You are a skilled writer who transforms raw research into concise, "
                  "easy to read summaries for a general audience.",
        llm=llm,
        verbose=True,
    )

    fact_checker = Agent(
        role="Fact Checker",
        goal="Verify that the summary report is accurately grounded in the research findings",
        backstory="You are a careful editor who cross checks written summaries against the original "
                  "research to catch unsupported claims, and produces a final, verified version.",
        llm=llm,
        verbose=True,
    )

    research_extra = f" Additional focus requested by the user: {research_notes.strip()}" if research_notes.strip() else ""
    writing_extra = f" Additional instructions from the user: {writing_notes.strip()}" if writing_notes.strip() else ""

    research_task = Task(
        description="You must use the Web Search Tool to search for information about the topic: {topic}. "
                    "Do not answer from memory. Call the Web Search Tool first, then report the key facts "
                    f"you found from its results.{research_extra}",
        expected_output="A structured collection of key facts and information about the topic, based only "
                        "on the Web Search Tool's results.",
        agent=researcher,
    )

    writing_task = Task(
        description=f"Using the research findings, write a clear, well organized summary report "
                    f"of approximately {word_count} words about {{topic}}. Structure it with a brief intro, "
                    f"key points, and a short conclusion.{writing_extra}",
        expected_output=f"A polished, easy to read summary report of about {word_count} words in plain English.",
        agent=writer,
        context=[research_task],
    )

    fact_check_task = Task(
        description="Compare the summary report against the original research findings. Check every claim "
                    "in the report is actually supported by the research. If something is unsupported or "
                    "exaggerated, correct it. If everything checks out, say so briefly at the top, then "
                    "output the final, verified version of the report.",
        expected_output="A final verified report, with a short note at the top confirming whether any "
                        "corrections were made.",
        agent=fact_checker,
        context=[research_task, writing_task],
    )

    return Crew(
        agents=[researcher, writer, fact_checker],
        tasks=[research_task, writing_task, fact_check_task],
        process=Process.sequential,
        verbose=True,
    )


def render_home():
    st.markdown(
        """
        <div class="vc-hero">
            <div class="vc-hero-eyebrow">Multi agent research assistant</div>
            <h1 style="margin:0; padding:0;">VeriCrew</h1>
            <p style="color:#5B5B52; font-size:15.5px; margin-top:0.4rem; max-width:640px;">
                A Researcher agent searches the web, a Writer agent drafts a summary, and a
                Fact Checker agent verifies it before the final report is shown.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    topic = st.text_input("Enter any topic", placeholder="e.g. climate change, AI in healthcare, Mughal history").strip()
    word_count = st.slider("Target report length (words)", min_value=100, max_value=400, value=225, step=25)

    with st.expander("Give individual instructions to each agent (optional)"):
        research_notes = st.text_area(
            "Tell the Researcher agent what to focus on",
            placeholder="e.g. focus on recent developments from 2025-2026, include statistics if available",
        )
        writing_notes = st.text_area(
            "Tell the Writer agent how to write it",
            placeholder="e.g. use simpler language, add a bullet point list of pros and cons",
        )

    run_button = st.button("Run agent crew", type="primary")

    if run_button:
        if not topic:
            st.warning("Please enter a topic first.")
            return

        last_sources.clear()
        with st.spinner(f"Agents are researching, writing, and fact checking '{topic}'. This may take a few minutes on a local model."):
            crew = get_crew(word_count, research_notes, writing_notes)
            result = crew.kickoff(inputs={"topic": topic})
        sources_found = list(last_sources)

        st.success("Done")

        img_url, img_page = fetch_wikipedia_image(topic)
        if img_url:
            st.image(img_url, caption="Image via Wikipedia" + (f" - {img_page}" if img_page else ""), width=300)

        tasks_output = result.tasks_output
        research_output = tasks_output[0].raw if len(tasks_output) > 0 else "N/A"
        draft_report = tasks_output[1].raw if len(tasks_output) > 1 else "N/A"
        final_report = tasks_output[2].raw if len(tasks_output) > 2 else result.raw

        st.markdown('<span class="vc-badge vc-badge-research">Agent 1 &middot; Research Specialist</span>', unsafe_allow_html=True)
        st.caption("Searched the web and gathered raw findings")
        with st.container(border=True):
            st.write(research_output)

        if sources_found:
            st.markdown("**Sources found by the Researcher agent**")
            seen_urls = set()
            for src in sources_found:
                if src["url"] not in seen_urls:
                    seen_urls.add(src["url"])
                    st.markdown(f'<div class="vc-source-link">- <a href="{src["url"]}" target="_blank">{src["title"]}</a></div>', unsafe_allow_html=True)

        kw_fig = plot_keyword_frequency(research_output)
        if kw_fig:
            st.pyplot(kw_fig)

        st.markdown('<span class="vc-badge vc-badge-writer">Agent 2 &middot; Content Writer</span>', unsafe_allow_html=True)
        st.caption("Turned the research into a draft summary")
        with st.container(border=True):
            st.write(draft_report)

        st.markdown('<span class="vc-badge vc-badge-verified">Agent 3 &middot; Fact Checker <span class="vc-stamp">Verified</span></span>', unsafe_allow_html=True)
        st.caption("Cross checked the draft against the research and produced the final version")
        with st.container(border=True):
            st.write(final_report)

        st.divider()
        try:
            pdf_bytes = generate_pdf(topic, research_output, final_report, sources_found)
            st.download_button(
                label="Download full report as PDF",
                data=pdf_bytes,
                file_name=f"research_report_{topic.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Could not generate PDF ({e}). You can still copy the text above manually.")


def render_about():
    st.markdown(
        """
        <div class="vc-hero">
            <div class="vc-hero-eyebrow">Project background</div>
            <h1 style="margin:0; padding:0;">About VeriCrew</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("The idea")
    st.write(
        "Most AI chat tools answer a question in one pass: one model does the retrieval, "
        "the reasoning, and the writing all at once, with no internal checkpoint to catch "
        "a mistake before it reaches the user. VeriCrew separates that single pass into three "
        "specialized agents that hand work off to each other, the way a small research team would: "
        "one person finds the sources, one person writes it up, and one person checks the write-up "
        "against the sources before it goes out."
    )

    st.subheader("How it works")
    st.markdown(
        "1. **Research Specialist** receives the topic and is required to call a live web search "
        "tool. It is not allowed to answer from memory alone, so its output is grounded in current, "
        "real sources rather than the model's training data.\n\n"
        "2. **Content Writer** receives the Researcher's findings and drafts a structured summary "
        "at the length the user requested.\n\n"
        "3. **Fact Checker** receives both the original research and the draft, checks each claim "
        "in the draft against the research, corrects anything unsupported, and produces the final "
        "version shown to the user.\n\n"
        "Each agent's output is shown separately in the app, along with the real source links, a "
        "topic image, and a keyword chart, so the process stays visible rather than hidden inside "
        "one black box answer."
    )

    st.subheader("Technology used")
    st.markdown(
        "- CrewAI for multi agent orchestration\n"
        "- Ollama running Llama 3.2 locally, so the project needs no paid API key\n"
        "- DuckDuckGo search for live web grounding\n"
        "- Streamlit for this interface\n"
        "- The Wikipedia REST API for topic images\n"
        "- Matplotlib for the keyword frequency chart\n"
        "- FPDF2 for the downloadable PDF report"
    )

    st.subheader("Known limitations")
    st.markdown(
        "- The local model is small, so it occasionally skips a tool call or repeats itself. "
        "A larger model would be more reliable.\n"
        "- The Fact Checker's corrections have not been benchmarked against a labeled dataset, "
        "so its accuracy is not formally measured.\n"
        "- Search results come from a single free source and are not ranked by credibility."
    )

    st.subheader("Credits")
    st.markdown(
        "Built by Juveria Yameen, M.Tech Artificial Intelligence and Data Science, "
        "Dr. V.R.K. Women's College of Engineering and Technology, Hyderabad.\n\n"
        "Built with CrewAI, Ollama, Streamlit, DuckDuckGo Search, the Wikipedia API, "
        "Matplotlib, and FPDF2."
    )


if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown(
        """
        <div class="vc-logo-row">
            <div class="vc-logo-mark">VC</div>
            <div>
                <div class="vc-logo-text-main">VeriCrew</div>
                <div class="vc-logo-text-sub">Research &amp; verification</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="vc-nav-label">Navigate</div>', unsafe_allow_html=True)

    if st.button("Home", use_container_width=True, type="primary" if st.session_state.page == "Home" else "secondary"):
        st.session_state.page = "Home"
        st.rerun()
    if st.button("About", use_container_width=True, type="primary" if st.session_state.page == "About" else "secondary"):
        st.session_state.page = "About"
        st.rerun()

    st.markdown(
        """
        <div class="vc-sidebar-footer">
            <div class="vc-status-row"><span class="vc-status-dot"></span> Local model: Llama 3.2 (1B)</div>
            <div class="vc-status-row"><span class="vc-status-dot"></span> Search: DuckDuckGo (live)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.page
if page == "Home":
    render_home()
else:
    render_about()