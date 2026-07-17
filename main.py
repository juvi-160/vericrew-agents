
from dotenv import load_dotenv
import os
from ddgs import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

load_dotenv()  # Load environment variables from .env file
# os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# ---- Custom tool: free web search using DuckDuckGo, works for ANY topic ----
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Searches the web for a given query/topic and returns a summary of top results.
    Use this to research any topic - news, facts, concepts, current events, etc."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return f"No results found for '{query}'."
        formatted = f"Search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            formatted += f"{i}. {r.get('title', 'No title')}\n{r.get('body', 'No description')}\nSource: {r.get('href', '')}\n\n"
        return formatted
    except Exception as e:
        return f"Search failed: {str(e)}"

word_count = 225

# ---- LLM config: local Ollama model, no API key needed ----
llm = LLM(
    model="openai/llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=max(300, int(word_count * 1.8))
)
# ---- Agents ----
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

# ---- Tasks ----
research_task = Task(
    description="Search the web for information about the topic: {topic}. "
                "Gather the most relevant and important facts, using the Web Search Tool.",
    expected_output="A structured collection of key facts and information about the topic.",
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

# ---- Crew ----
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    topic = input("Enter any topic to research (e.g. 'climate change', 'AI in healthcare'): ").strip()
    result = crew.kickoff(inputs={"topic": topic})
    print("\n\n===== FINAL REPORT =====\n")
    print(result.raw)