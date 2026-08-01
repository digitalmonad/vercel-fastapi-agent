import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_cloudflare import ChatCloudflareWorkersAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge-base.json"

ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
API_KEY = os.getenv("CLOUDFLARE_API_KEY")
MODEL = os.getenv("CLOUDFLARE_MODEL")

llm = ChatCloudflareWorkersAI(
    model=MODEL,
    account_id=ACCOUNT_ID,
    api_token=API_KEY,
)


with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as handle:
    KNOWLEDGE_BASE = json.load(handle)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for relevant information."""
    query_lower = query.lower()
    matches = []

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                walk(child, new_prefix)
        elif isinstance(value, list):
            for item in value:
                walk(item, prefix)
        else:
            text = str(value).lower()
            if query_lower in text:
                matches.append(f"{prefix}: {value}")

    walk(KNOWLEDGE_BASE)

    if not matches:
        return "No matching information was found in the knowledge base."

    return "\n".join(matches[:8])


@tool
def get_company_summary() -> str:
    """Return a short summary of the company and its products."""
    company = KNOWLEDGE_BASE.get("company", {})
    products = KNOWLEDGE_BASE.get("products", [])
    summary_lines = [
        f"Company: {company.get('name', 'Unknown')}",
        f"Industry: {company.get('industry', 'Unknown')}",
        f"Mission: {company.get('mission', 'Unknown')}",
    ]
    if products:
        summary_lines.append("Products:")
        for product in products:
            summary_lines.append(
                f"- {product.get('name', 'Unknown')}: {product.get('description', '')}"
            )
    return "\n".join(summary_lines)


TOOLS = [search_knowledge_base, get_company_summary]

SYSTEM_MESSAGE = (
    "You are a helpful bot to answer frequently asked questions. "
    "You answer users questions ONLY about our company. "
    "Be concise and helpful."
)

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)


def run_agent(user_input: str) -> str:
    """Run the agent with a user query and return the response."""
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={
                "recursion_limit": 5
            },  # agent can do only limited number of iterations
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"
