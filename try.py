import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
#google_genai:gemini-3.5-flash-lite

from langchain.agents import create_agent

import asyncio
from langchain.agents import create_agent
#from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from toolbox_langchain import ToolboxClient

prompt = """
You're a helpful hotel assistant. You handle hotel searching and booking.
When the user searches for a hotel, list the full details for each hotel found: id, name, location, and price tier.
Always use the hotel ID for booking operations.
For any bookings, provide a clear confirmation message.
Don't ask for clarification or confirmation from the user; perform the requested action directly.
"""


async def run_queries(agent_executor):
    config = {"configurable": {"thread_id": "hotel-thread-1"}}

    # --- Query 1: Search for hotels ---
    query1 = "I need to find a hotel in Basel."
    print(f'\n--- USER: "{query1}" ---')
    inputs1 = {"messages": [("user", prompt + query1)]}
    async for event in agent_executor.astream_events(
        inputs1, config=config, version="v2"
    ):
        if event["event"] == "on_chat_model_end" and event["data"]["output"].content:
            print(f"--- AGENT: ---\n{event['data']['output'].content}")

    # --- Query 2: Book a hotel ---
    query2 = "Great, please book the Hyatt Regency Basel for me."
    print(f'\n--- USER: "{query2}" ---')
    inputs2 = {"messages": [("user", query2)]}
    async for event in agent_executor.astream_events(
        inputs2, config=config, version="v2"
    ):
        if event["event"] == "on_chat_model_end" and event["data"]["output"].content:
            print(f"--- AGENT: ---\n{event['data']['output'].content}")


async def run_hotel_agent():
    model = "google_genai:gemini-3.5-flash-lite"

    # Load the tools from the running MCP Toolbox server
    async with ToolboxClient("http://127.0.0.1:5000") as client:
        tools = await client.aload_toolset("hotel_toolset")
        agent = create_agent(model, tools, checkpointer=MemorySaver())
        await run_queries(agent)


async def main():
    await run_hotel_agent()


asyncio.run(main())