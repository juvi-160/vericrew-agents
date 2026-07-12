import streamlit as st
from ddgs import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from fpdf import FPDF
import io
import textwrap
import requests
import re
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Research Crew", page_icon="🔎", layout="centered")

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
    ax.barh(labels[::-1], values[::-1], color="#4C72B0")
    ax.set_xlabel("Mentions")
    ax.set_title("Most Frequent Keywords in Research Findings")
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


# Stores the last set of search sources so the UI can show real clickable links
# separately from what the LLM reads/summarizes
last_sources = []


# ---- Custom tool: free web search using DuckDuckGo, works for ANY topic ----
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Searches the web for a given query/topic and returns a summary of top results.
    Use this to research any topic - news, facts, concepts, current events, etc."""
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
    llm = LLM(model="ollama/llama3.2:1b", base_url="http://localhost:11434", max_tokens=max(300, int(word_count * 1.8)))

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
        goal="Turn research findings into a clear, well-organized summary report",
        backstory="You are a skilled writer who transforms raw research into concise, "
                  "easy-to-read summaries for a general audience.",
        llm=llm,
        verbose=True,
    )

    fact_checker = Agent(
        role="Fact Checker",
        goal="Verify that the summary report is accurately grounded in the research findings",
        backstory="You are a careful editor who cross-checks written summaries against the original "
                  "research to catch unsupported claims, and produces a final, verified version.",
        llm=llm,
        verbose=True,
    )

    research_extra = f" Additional focus requested by the user: {research_notes.strip()}" if research_notes.strip() else ""
    writing_extra = f" Additional instructions from the user: {writing_notes.strip()}" if writing_notes.strip() else ""

    research_task = Task(
        description="You MUST use the Web Search Tool to search for information about the topic: {topic}. "
                    "Do not answer from memory. Call the Web Search Tool first, then report the key facts "
                    f"you found from its results.{research_extra}",
        expected_output="A structured collection of key facts and information about the topic, based only "
                        "on the Web Search Tool's results.",
        agent=researcher,
    )

    writing_task = Task(
        description=f"Using the research findings, write a clear, well-organized summary report "
                    f"of approximately {word_count} words about {{topic}}. Structure it with a brief intro, "
                    f"key points, and a short conclusion.{writing_extra}",
        expected_output=f"A polished, easy-to-read summary report of about {word_count} words in plain English.",
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


def generate_pdf(topic, research_text, report_text, sources=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    add_wrapped_text(pdf, f"AI Research Crew Report: {topic}", width_chars=70, line_height=9)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    add_wrapped_text(pdf, "Research Findings (Researcher Agent)", width_chars=90, line_height=7)
    pdf.set_font("Helvetica", "", 10)
    add_wrapped_text(pdf, research_text)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    add_wrapped_text(pdf, "Summary Report (Writer Agent)", width_chars=90, line_height=7)
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
                add_wrapped_text(pdf, f"- {src['title']}: {src['url']}")

    return bytes(pdf.output())


# ---- UI ----
st.title("🔎 AI Research Crew")
st.caption("Multi-agent system: a Researcher agent searches the web on any topic, "
           "then a Writer agent turns it into a summary report — powered by local CrewAI + Ollama.")

topic = st.text_input("Enter any topic", placeholder="e.g. climate change, AI in healthcare, Mughal history").strip()
word_count = st.slider("Target report length (words)", min_value=100, max_value=400, value=225, step=25)

with st.expander("⚙️ Give individual instructions to each agent (optional)"):
    research_notes = st.text_area(
        "Tell the Researcher agent what to focus on",
        placeholder="e.g. focus on recent developments from 2025-2026, include statistics if available",
    )
    writing_notes = st.text_area(
        "Tell the Writer agent how to write it",
        placeholder="e.g. use simpler language, add a bullet-point list of pros and cons",
    )

run_button = st.button("Run Agent Crew", type="primary")

if run_button:
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        last_sources.clear()
        with st.spinner(f"Agents are researching, writing, and fact-checking '{topic}'... this may take a few minutes on a local model."):
            crew = get_crew(word_count, research_notes, writing_notes)
            result = crew.kickoff(inputs={"topic": topic})
        sources_found = list(last_sources)

        st.success("Done!")

        # Wikipedia thumbnail for the topic, if one exists
        img_url, img_page = fetch_wikipedia_image(topic)
        if img_url:
            st.image(img_url, caption=f"Image via Wikipedia" + (f" — {img_page}" if img_page else ""), width=300)

        # Pull individual task outputs so we can show each agent's contribution separately
        tasks_output = result.tasks_output
        research_output = tasks_output[0].raw if len(tasks_output) > 0 else "N/A"
        draft_report = tasks_output[1].raw if len(tasks_output) > 1 else "N/A"
        final_report = tasks_output[2].raw if len(tasks_output) > 2 else result.raw

        st.markdown("### 🕵️ Agent 1: Research Specialist")
        st.caption("Searched the web and gathered raw findings")
        with st.container(border=True):
            st.write(research_output)

        if sources_found:
            st.markdown("**🔗 Sources found by the Researcher agent:**")
            seen_urls = set()
            for src in sources_found:
                if src["url"] not in seen_urls:
                    seen_urls.add(src["url"])
                    st.markdown(f"- [{src['title']}]({src['url']})")

        kw_fig = plot_keyword_frequency(research_output)
        if kw_fig:
            st.pyplot(kw_fig)

        st.markdown("### ✍️ Agent 2: Content Writer")
        st.caption("Turned the research into a draft summary")
        with st.container(border=True):
            st.write(draft_report)

        st.markdown("### ✅ Agent 3: Fact Checker")
        st.caption("Cross-checked the draft against the research and produced the final version")
        with st.container(border=True):
            st.write(final_report)

        st.divider()
        try:
            pdf_bytes = generate_pdf(topic, research_output, final_report, sources_found)
            st.download_button(
                label="📄 Download Full Report as PDF",
                data=pdf_bytes,
                file_name=f"research_report_{topic.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Couldn't generate PDF ({e}). You can still copy the text above manually.")