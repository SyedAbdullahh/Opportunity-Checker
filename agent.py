import json
import os
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from db import get_user_profile, save_user_profile
from tools import scan_gmail_for_opportunities, store_profile_from_message


def build_agent(user_email: str, credentials_data: dict[str, Any], chat_history: list[dict[str, str]]):
    disable_ssl_verify = os.getenv("OPENAI_DISABLE_SSL_VERIFY", "true").lower() in {"1", "true", "yes", "on"}
    api_key = os.getenv("OPENAI_API_KEY")

    def get_profile_tool() -> str:
        profile = get_user_profile(user_email)
        return json.dumps({"found": bool(profile), "profile": profile}, ensure_ascii=False)

    def save_profile_tool(profile_json: str) -> str:
        try:
            profile_data = json.loads(profile_json)
        except json.JSONDecodeError:
            profile_data = store_profile_from_message(user_email, profile_json)
        else:
            save_user_profile(user_email, profile_data)
        return json.dumps({"stored": True, "user_email": user_email}, ensure_ascii=False)

    def scan_opportunities_tool(max_results: int = 30) -> str:
        profile = get_user_profile(user_email)
        result = scan_gmail_for_opportunities(
            user_email=user_email,
            credentials_data=credentials_data,
            profile_data=profile,
            max_results=max_results,
        )
        return json.dumps(result, ensure_ascii=False)

    tools = [
        StructuredTool.from_function(
            func=get_profile_tool,
            name="get_user_profile",
            description="Get the logged-in user's saved profile from SQLite. Returns JSON with found/profile.",
        ),
        StructuredTool.from_function(
            func=save_profile_tool,
            name="save_user_profile",
            description="Save a structured JSON profile for the logged-in user into SQLite.",
        ),
        StructuredTool.from_function(
            func=scan_opportunities_tool,
            name="scan_opportunities",
            description="Fetch opportunity-related Gmail messages, score them, and return ranked results as JSON.",
        ),
    ]

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    http_client = httpx.Client(verify=not disable_ssl_verify)
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        http_client=http_client,
        api_key=api_key,
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are Opportunity Inbox Copilot. First check if the user profile exists by calling get_user_profile. "
            "If missing, ask the user for all profile fields in a compact list: degree_program, semester, cgpa, skills_interests, "
            "preferred_opportunity_types, financial_need, location_preference, past_experience. When the user provides the details, "
            "call save_user_profile with valid JSON. After a profile exists, call scan_opportunities to retrieve Gmail opportunities. "
            "Always explain what you did and present the results clearly. Never invent profile data or email results."
        ),
    )

    history_messages: list[Any] = []
    for message in chat_history:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    return agent, history_messages
