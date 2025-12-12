import datetime
import logging
import os

from crewai import Agent, Crew, Task
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

from ut_openai_usage import get_openai_usage
from ut_tavily_usage import get_tavily_usage

logging.basicConfig(level=logging.INFO)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY environment variable is not set.")

tavily_tool = TavilySearchTool()
today = datetime.date.today().isoformat()
date_tag = f"(Generated on {today})"

# Agent 1: Current Bitcoin Market Analyst
agent1 = Agent(
    role="Current Bitcoin Market Analyst",
    goal="Assess the current Bitcoin market and identify signs of coordinated influence or manipulation.",
    backstory="A crypto market analyst who specializes in detecting unusual price behavior, sentiment shifts, and potential behind-the-scenes actors shaping Bitcoin’s trajectory.",
    tools=[tavily_tool],
    verbose=True,
)

task1 = Task(
    description=f"{date_tag} Analyze the current Bitcoin market status. Identify any signs of manipulation, coordinated influence, or unusual volatility. Look for actors or institutions that may have benefited from recent price moves. Cite all sources clearly (names, institutions, publications, websites).",
    agent=agent1,
    expected_output="Markdown-formatted summary of current Bitcoin market status with cited sources and speculative signals.",
)

# Agent 2: 1-Month Retrospective Researcher
agent2 = Agent(
    role="1-Month Bitcoin Conspiracy Researcher",
    goal="Investigate Bitcoin-related events from one month ago that may have been part of a coordinated effort to influence today’s market.",
    backstory="A researcher trained to detect subtle signals, accumulation patterns, or media narratives that may have been seeded to manipulate Bitcoin’s price.",
    tools=[tavily_tool],
    verbose=True,
)

task2 = Task(
    description=f"{date_tag} Search for Bitcoin-related news and signals from around one month ago. Look for signs of strategic accumulation, narrative planting, or influential actors positioning themselves. For example, if a political figure’s family launched a Bitcoin-related company, consider how that may have influenced price. Cite all sources clearly (names, institutions, publications, websites).",
    agent=agent2,
    expected_output="Markdown-formatted summary of 1-month-old signals with cited sources and speculative interpretation.",
)

# Agent 3: 3-Month Retrospective Researcher
agent3 = Agent(
    role="3-Month Bitcoin Conspiracy Researcher",
    goal="Investigate Bitcoin-related events from three months ago that may have been part of a coordinated effort to influence today’s market.",
    backstory="A researcher trained to detect subtle signals, accumulation patterns, or media narratives that may have been seeded to manipulate Bitcoin’s price.",
    tools=[tavily_tool],
    verbose=True,
)

task3 = Task(
    description=f"{date_tag} Search for Bitcoin-related news and signals from around three months ago. Look for signs of strategic accumulation, narrative planting, or influential actors positioning themselves. For example, if a political figure’s family launched a Bitcoin-related company, consider how that may have influenced price. Cite all sources clearly (names, institutions, publications, websites).",
    agent=agent3,
    expected_output="Markdown-formatted summary of 3-month-old signals with cited sources and speculative interpretation.",
)

# Agent 4: 6-Month Retrospective Researcher
agent4 = Agent(
    role="6-Month Bitcoin Conspiracy Researcher",
    goal="Investigate Bitcoin-related events from six months ago that may have been part of a coordinated effort to influence today’s market.",
    backstory="A researcher trained to detect subtle signals, accumulation patterns, or media narratives that may have been seeded to manipulate Bitcoin’s price.",
    tools=[tavily_tool],
    verbose=True,
)

task4 = Task(
    description=f"{date_tag} Search for Bitcoin-related news and signals from around six months ago. Look for signs of strategic accumulation, narrative planting, or influential actors positioning themselves. For example, if a political figure’s family launched a Bitcoin-related company, consider how that may have influenced price. Cite all sources clearly (names, institutions, publications, websites).",
    agent=agent4,
    expected_output="Markdown-formatted summary of 6-month-old signals with cited sources and speculative interpretation.",
)

# Agent 5: Strategic Conspiracy Analyst
agent5 = Agent(
    role="Bitcoin Conspiracy-Based Strategist",
    goal="Synthesize multi-period Bitcoin signals to uncover hidden agendas and forecast short-term market moves.",
    backstory="A strategist who connects dots across time, embracing conspiracy theories and speculative synthesis to expose coordinated market manipulation. For example, if Trump’s son launched a Bitcoin accumulation firm, consider how presidential influence could shape future price moves.",
    tools=[tavily_tool],
    verbose=True,
)

task5 = Task(
    description=f"""{date_tag} Using insights from Agents 1~4, speculate on how the Bitcoin market may evolve over the next 1~3 months. Focus on suspicious patterns, coordinated narratives, and conspiracy-like signals. For example, if Trump’s son launched a Bitcoin accumulation firm and Trump later made statements that crashed the market, consider how this may have been orchestrated. Final output must be written in Korean and formatted in Markdown. Cite all sources clearly (names, institutions, publications, websites).""",
    agent=agent5,
    expected_output="Markdown-formatted Korean report with speculative forecast, conspiracy signals, and cited sources.",
    context=[task1, task2, task3, task4],
)

# Crew setup
crew = Crew(
    agents=[agent1, agent2, agent3, agent4, agent5],
    tasks=[task1, task2, task3, task4, task5],
    verbose=True,
)

result = crew.kickoff()

print("====================================")
print("📄 Final Markdown Report (in Korean):")
print(result)

print("OpenAI Usage: ", get_openai_usage())
print("Tavily Usage: ", get_tavily_usage())
