import os
import time
from crewai import Agent, Task, Crew, Process, LLM

# Read your API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Use Gemini 2.5 Flash model
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.0,
)

# Define the Agent
greeter = Agent(
    role="Greeter",
    goal="Create a friendly greeting for the user.",
    backstory="You are a friendly AI assistant who loves to greet people.",
    llm=gemini_llm,
    verbose=True,
)

# Define the Task
greeting_task = Task(
    description="Create a greeting for a user named {name}.",
    expected_output="A friendly greeting string.",
    agent=greeter,
)

# Define the Crew with caching enabled
crew = Crew(
    agents=[greeter],
    tasks=[greeting_task],
    process=Process.sequential,
    cache=True,  # Enable caching
    verbose=True,
)

# Inputs
inputs = {"name": "Venkman"}

print("--- Run Started ---")
start_time = time.time()
result = crew.kickoff(inputs=inputs)
end_time = time.time()
print(f"Result: {result}")
print(f"Time taken: {end_time - start_time:.4f} seconds")
print(f"Token Usage: {result.token_usage}")
