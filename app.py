import streamlit as st
from ddgs import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from fpdf import FPDF
import io
import re

st.set_page_config(page_title="AI Research Crew", page_icon="🔎", layout="centered")


def safe_wrap(text, max_word_len=50):
    """Breaks up very long unbroken strings (like URLs) so FPDF can wrap them onto new lines."""
    if not text:
        return ""

    def breaker(match):
        word = match.group(0)
        return " ".join(word[i:i + max_word_len] for i in range(0, len(word), max_word_len))

    return re.sub(r"\S{%d,}" % (max_word_len + 1), breaker, text)


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
def get_crew():
    llm = LLM(model="ollama/llama3.2:1b", base_url="http://localhost:11434", max_tokens=400)

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

    research_task = Task(
        description="You MUST use the Web Search Tool to search for information about the topic: {topic}. "
                    "Do not answer from memory. Call the Web Search Tool first, then report the key facts "
                    "you found from its results.",
        expected_output="A structured collection of key facts and information about the topic, based only "
                        "on the Web Search Tool's results.",
        agent=researcher,
    )

    writing_task = Task(
        description="Using the research findings, write a clear, well-organized summary report "
                    "(200-250 words) about {topic}. Structure it with a brief intro, key points, "
                    "and a short conclusion.",
        expected_output="A polished, easy-to-read summary report in plain English.",
        agent=writer,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )


def generate_pdf(topic, research_text, report_text, sources=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, safe_wrap(f"AI Research Crew Report: {topic}"))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, "Research Findings (Researcher Agent)")
    pdf.set_font("Helvetica", "", 10)
    clean_research = safe_wrap(research_text.encode("latin-1", "replace").decode("latin-1"))
    pdf.multi_cell(0, 6, clean_research if clean_research.strip() else "(No content returned by agent)")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, "Summary Report (Writer Agent)")
    pdf.set_font("Helvetica", "", 10)
    clean_report = safe_wrap(report_text.encode("latin-1", "replace").decode("latin-1"))
    pdf.multi_cell(0, 6, clean_report if clean_report.strip() else "(No content returned by agent)")

    if sources:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, "Sources")
        pdf.set_font("Helvetica", "", 10)
        seen = set()
        for src in sources:
            if src["url"] not in seen:
                seen.add(src["url"])
                line = safe_wrap(f"- {src['title']}: {src['url']}")
                pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"))

    return bytes(pdf.output())


# ---- UI ----
st.title("🔎 AI Research Crew")
st.caption("Multi-agent system: a Researcher agent searches the web on any topic, "
           "then a Writer agent turns it into a summary report — powered by local CrewAI + Ollama.")

topic = st.text_input("Enter any topic", placeholder="e.g. climate change, AI in healthcare, Mughal history").strip()
run_button = st.button("Run Agent Crew", type="primary")

if run_button:
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        last_sources.clear()
        with st.spinner(f"Agents are researching and writing about '{topic}'... this may take a minute or two on a local model."):
            crew = get_crew()
            result = crew.kickoff(inputs={"topic": topic})
        sources_found = list(last_sources)

        st.success("Done!")

        # Pull individual task outputs so we can show each agent's contribution separately
        tasks_output = result.tasks_output
        research_output = tasks_output[0].raw if len(tasks_output) > 0 else "N/A"
        report_output = tasks_output[1].raw if len(tasks_output) > 1 else result.raw

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

        st.markdown("### ✍️ Agent 2: Content Writer")
        st.caption("Turned the research into a polished summary")
        with st.container(border=True):
            st.write(report_output)

        st.divider()
        try:
            pdf_bytes = generate_pdf(topic, research_output, report_output, sources_found)
            st.download_button(
                label="📄 Download Full Report as PDF",
                data=pdf_bytes,
                file_name=f"research_report_{topic.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Couldn't generate PDF ({e}). You can still copy the text above manually.")