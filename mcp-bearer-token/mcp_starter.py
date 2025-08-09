import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
from io import StringIO

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
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
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
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
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

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
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []
        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    # auth=SimpleBearerAuthProvider(TOKEN),
)

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
    job_description: Annotated[str | None, Field(description="Full job description text")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    if job_description:
        return f"📝 **Job Description Analysis**\n\n---\n{job_description.strip()}\n---\n\nUser Goal: **{user_goal}**"

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n---\n{content.strip()}\n---"

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return f"🔍 **Search Results for**: _{user_goal}_\n\n" + "\n".join(f"- {link}" for link in links)

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide job description, job URL, or search query."))

# --- Tool: make_img_black_and_white ---
MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use when the user provides an image URL to convert to black and white.",
    side_effects="The image will be processed and saved in black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data")] = None,
) -> list[TextContent | ImageContent]:
    import base64, io
    from PIL import Image
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

# --- Tool: stock_predictor (Puch AI) ---
STOCK_PREDICTOR_DESCRIPTION = RichToolDescription(
    description="NSE stock prediction and recommendation using Puch AI multi-agent workflow.",
    use_when="Use when the user wants Indian NSE stock recommendations or predictions.",
    side_effects="Fetches market data, analyzes news, and returns buy/sell suggestions.",
)

@mcp.tool(description=STOCK_PREDICTOR_DESCRIPTION.model_dump_json())
async def stock_predictor(
    user_query: Annotated[str, Field(description="Stock-related request")]
) -> str:
    # ✅ Imports
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor
    from langchain_core.messages import convert_to_messages

    output_buf = StringIO()

    def pretty_print_messages(update, buf, last_message=False):
        """Format and print messages from agents."""
        if isinstance(update, tuple):
            ns, update = update
            if len(ns) == 0:
                return
        for node_name, node_update in update.items():
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]
            for m in messages:
                buf.write(m.pretty_repr(html=False) + "\n\n")

    # ✅ MCP client setup
    client = MultiServerMCPClient({
        "bright_data": {
            "command": "npx",
            "args": ["@brightdata/mcp"],
            "env": {
                "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
            },
            "transport": "stdio",
        },
    })

    tools = await client.get_tools()

    # ✅ Gemini model with timeout & smaller output size
    def gemini_model():
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            max_output_tokens=400,  # limit response size
            request_timeout=5        # 5 sec per call
        )

    # ✅ Agents with concise prompts
    stock_finder_agent = create_react_agent(
        gemini_model(), tools,
        prompt="Pick 2 actively traded NSE stocks (no penny stocks) for short-term trading. Give ticker, name, and brief reason.",
        name="stock_finder_agent"
    )

    market_data_agent = create_react_agent(
        gemini_model(), tools,
        prompt="For given NSE tickers, provide: current price, prev close, today's volume, 7d/30d trend, RSI, 50/200 MA, notable volume/volatility spikes.",
        name="market_data_agent"
    )

    news_analyst_agent = create_react_agent(
        gemini_model(), tools,
        prompt="For NSE tickers, summarize latest 3-5 days news with sentiment (positive/negative/neutral) and short-term price impact.",
        name="news_analyst_agent"
    )

    price_recommender_agent = create_react_agent(
        gemini_model(), tools,
        prompt="Given market data + news, recommend Buy/Sell/Hold with target price (INR) and 1-line reason.",
        name="price_recommender_agent"
    )

    # ✅ Supervisor in parallel mode
    supervisor = create_supervisor(
        model=gemini_model(),
        agents=[
            stock_finder_agent,
            market_data_agent,
            news_analyst_agent,
            price_recommender_agent
        ],
        prompt="Coordinate agents to deliver actionable NSE stock recommendations quickly.",
        add_handoff_back_messages=True,
        output_mode="full_history",
        parallel=True
    ).compile()

    # ✅ Run with global 20-second limit
    async def run_with_timeout():
        for chunk in supervisor.stream({"messages": [{"role": "user", "content": user_query}]}):
            pretty_print_messages(chunk, output_buf, last_message=True)

    try:
        await asyncio.wait_for(run_with_timeout(), timeout=20)
    except asyncio.TimeoutError:
        output_buf.write("\n⚠️ Process timed out after 20 seconds. Partial results shown.\n")

    return output_buf.getvalue().strip()

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
