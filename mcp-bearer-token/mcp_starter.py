import asyncio
import os
import base64
import io
import random
import markdownify
from langchain.schema import SystemMessage, HumanMessage
import requests
import pandas as pd
import httpx
import readabilipy
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field, AnyUrl
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages

from PIL import Image
from bs4 import BeautifulSoup


# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
)


# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None


# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls, url: str, user_agent: str, force_raw: bool = False
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"
                    )
                )

            if response.status_code >= 400:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to fetch {url} - status code {response.status_code}",
                    )
                )

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def Google_Search_links(query: str, num_results: int = 5) -> list[str]:
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []
        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]


# --- MCP Server Setup ---
mcp = FastMCP("Job Finder & Stock MCP Server")


# --- Tool: validate ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


# --- Tool: job_finder ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)


@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal")],
    job_description: Annotated[
        str | None, Field(description="Full job description text")
    ] = None,
    job_url: Annotated[
        AnyUrl | None, Field(description="A URL to fetch a job description from.")
    ] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    if job_description:
        return f"📝 **Job Description Analysis**\n\n---\n{job_description.strip()}\n---\n\nUser Goal: **{user_goal}**"

    if job_url:
        content, _ = await Fetch.fetch_url(
            str(job_url), Fetch.USER_AGENT, force_raw=raw
        )
        return f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n---\n{content.strip()}\n---"

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.Google_Search_links(user_goal)
        return f"🔍 **Search Results for**: _{user_goal}_\n\n" + "\n".join(
            f"- {link}" for link in links
        )

    raise McpError(
        ErrorData(
            code=INVALID_PARAMS,
            message="Provide job description, job URL, or search query.",
        )
    )


# --- Tool: make_img_black_and_white ---
MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use when the user provides an image URL to convert to black and white.",
    side_effects="The image will be processed and saved in black and white format.",
)


@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[
        str, Field(description="Base64-encoded image data")
    ] = None,
) -> list[TextContent | ImageContent]:
    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))
        bw_image = image.convert("L")
        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")
        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# --- Tool: stock_recommendation ---
STOCK_PREDICTOR_DESCRIPTION = RichToolDescription(
    description="NSE stock prediction and recommendation using Gemini multi-agent workflow.",
    use_when="Use when the user wants Indian NSE stock recommendations or predictions based on a news article. Provides risk, trend, and price targets.",
    side_effects="Fetches market data from the article, analyzes it, and returns a detailed financial report for a specific NSE stock ticker.",
)


def pretty_print_message(message, indent=False):
    return message.pretty_repr(html=True)


def extract_last_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return []
    results = []
    for _, node_update in update.items():
        messages = convert_to_messages(node_update["messages"])
        results.extend([pretty_print_message(m) for m in messages[-1:]])
    return results


def fetch_stock_news(stock_name, days=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": stock_name,
        "from": (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "apiKey": os.getenv("NEWSAPI_KEY"),
        "language": "en",
        "source" : "India, ind, Bharat, NSE, BSE"
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return []
    articles = resp.json().get("articles", [])
    return [
        {
            "title": a["title"],
            "description": a.get("description", ""),
            "url": a["url"],
            "publishedAt": a["publishedAt"],
            "source": a["source"]["name"],
        }
        for a in articles
    ]


async def run_stock_agent(query: str):
    import json
    from langchain.schema import SystemMessage, HumanMessage

    def build_company_extraction_user_prompt(q: str) -> str:
        hints = (
            "- If the query is generic (e.g., 'what stock to buy today', 'best stocks', sector-only), return NONE.\n"
            "- If multiple companies are mentioned, pick the most central one.\n"
            "- Prefer NSE-listed company names or well-known tickers."
        )
        return f"Query: {q}\nInstructions: Identify a single NSE-listed company if present; else return NONE.\n{hints}"

    COMPANY_EXTRACTION_SYSTEM = (
        "You extract a single NSE-listed company name mentioned in a short user query for Indian stocks. "
        "If the query is generic, return NONE. "
        "Respond ONLY with this JSON (no extra text): "
        '{ "company": "<Company or NONE>", "reason": "<brief reason>" }'
    )

    try:
        extraction_messages = [
            SystemMessage(content=COMPANY_EXTRACTION_SYSTEM),
            HumanMessage(content=build_company_extraction_user_prompt(query)),
        ]

        # Synchronous generate (can be converted to async if llm supports it)
        response = llm.generate([extraction_messages])
        content = response.generations[0][0].text.strip()

        data = json.loads(content)
        resolved = (data.get("company") or "").strip()
        reason = (data.get("reason") or "").strip()

    except Exception:
        resolved, reason = "NONE", "Extraction failed; defaulted to general NSE."

    target = resolved if resolved and resolved.upper() != "NONE" else "NSE"

    # Fetch recent news
    articles = fetch_stock_news(target, days=5)

    def fmt(a):
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        s = (a.get("source") or "").strip()
        p = (a.get("publishedAt") or "").strip()
        return f"- [{p}] {s}: {t} || {d}"

    news_block = "\n".join(fmt(a) for a in articles[:25]) or "No recent news found."

    tools = []
    stock_agent = create_react_agent(
        llm,
        tools,
        prompt=f"""
You are a single integrated stock research assistant specializing in the Indian Stock Market (NSE).
Handle the entire workflow in ONE go, without asking for follow-up approval.

Primary task:
- Use ONLY the provided recent news to infer near-term sentiment (Bullish/Bearish/Neutral) for the given stock.

Additional task:
- If the user query contains words like "buy", "recommend", "suggest", "best stocks", or "invest",
  then also recommend 2-3 NSE-listed stocks with strong short-term prospects based on recent news trends.
- Provide one short reason for each recommendation.

Resolved target from user query: {target}
Extraction note: {reason}

Recent News (last ~5 days) for: {target}
{news_block}

Tasks:
1. Classify the target's short-term sentiment as Bullish, Bearish, or Neutral based solely on the above news.
2. If applicable, suggest 2-3 other NSE stocks to buy, with one-line reasons.

Output format (plain text):
TARGET: {target}
SENTIMENT: <Bullish | Bearish | Neutral>
CONFIDENCE: <0-1>
REASON: <one short sentence referencing the news themes>
RECOMMENDATIONS (if applicable):
1. <Stock Name> - <reason>
2. <Stock Name> - <reason>
3. <Stock Name> - <reason>
""",
        name="stock_agent",
    )

    responses = []
    for chunk in stock_agent.stream({"messages": [{"role": "user", "content": query}]}):
        last_msgs = extract_last_messages(chunk)
        responses.extend(last_msgs)

    return "\n\n".join(responses)


@mcp.tool(description=STOCK_PREDICTOR_DESCRIPTION.model_dump_json())
async def stock_recommendation(query: str) -> str:
    return await run_stock_agent(query)


# --- Start MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)


if __name__ == "__main__":
    asyncio.run(main())
