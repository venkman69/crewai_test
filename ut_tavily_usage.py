import json
import os
from typing import Any

import requests
from dotenv import load_dotenv


def get_tavily_usage(api_key: str | None = None) -> dict[str, Any]:
    if not api_key:
        api_key = os.environ.get("TAVILY_API_KEY")

    USAGE_URL = "https://api.tavily.com/usage"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url=USAGE_URL, headers=headers)
        response.raise_for_status()
        usage_data = response.json()
        usage: dict = {
            "current_plan": usage_data["account"]["current_plan"],
            "plan_usage": usage_data["account"]["plan_usage"],
            "plan_limit": usage_data["account"]["plan_limit"],
        }
        return usage
    except Exception as e:
        print(f"Unknown exception occurred: {e}")
        return {}


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    usage_data: dict = get_tavily_usage(api_key)
    if usage_data:
        usage: dict = {
            "current_plan": usage_data["account"]["current_plan"],
            "plan_usage": usage_data["account"]["plan_usage"],
            "plan_limit": usage_data["account"]["plan_limit"],
        }
        print(json.dumps(usage))
    else:
        print("Failed to retrieve usage data.")
