import os
from crewai import Agent, Task, Crew, Process, LLM

# Read your API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Use Gemini 2.5 Flash model
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.0,
)


def create_crew():
    # Define the Agent
    greeter = Agent(
        role="Greeter",
        goal="Create a friendly greeting for the user.",
        backstory="You are a friendly AI assistant who loves to greet people.",
        llm=gemini_llm,
        verbose=True,
        id="00000000-0000-0000-0000-000000000001",
    )

    # Define the Task
    greeting_task = Task(
        description="Create a greeting for a user named {name}.",
        expected_output="A friendly greeting string.",
        agent=greeter,
        id="00000000-0000-0000-0000-000000000002",
    )

    # Define the Crew with caching enabled
    crew = Crew(
        agents=[greeter],
        tasks=[greeting_task],
        process=Process.sequential,
        cache=True,  # Enable caching
        verbose=True,
    )
    return crew, greeter, greeting_task


# Inputs
inputs = {"name": "Venkman"}

print("--- First Run ---")
crew1, greeter1, task1 = create_crew()
print(f"Agent ID: {greeter1.id}")
print(f"Task ID: {task1.id}")
result1 = crew1.kickoff(inputs=inputs)
print(f"Result 1 Token Usage: {result1.token_usage}")

print("\n--- Second Run (New Crew Instance) ---")
crew2, greeter2, task2 = create_crew()
print(f"Agent ID: {greeter2.id}")
print(f"Task ID: {task2.id}")
result2 = crew2.kickoff(inputs=inputs)
print(f"Result 2 Token Usage: {result2.token_usage}")
