import os

import dotenv
from langchain.tools import tool
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field
from typing import Literal
from langchain_community.tools import BraveSearch
from sqlalchemy.sql.functions import count

dotenv.load_dotenv()
brave_key = os.getenv("BRAVE_API")
class WeatherInput(BaseModel):
    """Input for weather query."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forcast: bool = Field(
        default=False,
        description="Whether to include a 3-day forecast"
    )


@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forcast: bool = False) -> str:
    """
    Get the current weather for a given location.

    Args:
        location (str): The city name or coordinates to get the weather for.
        units (str, optional): The temperature unit preference ("celsius" or "fahrenheit").
        include_forcast (bool, optional): Whether to include a 3-day forecast in the response.
    """
    return f"The current weather in {location} is 20 degrees {units}. Forecast included: {include_forcast}."

@tool
def search_news(query: str, limit: int = 10) -> str:
    """Search the news articles and summary.
    Args:
        query (str): The search query.
        limit (int, optional): The maximum number of news to return.
    """
    writer = get_stream_writer()
    writer(f"Searching news for {query}.")
    bravesearch = BraveSearch.from_api_key(api_key=brave_key, search_kwargs={"count": limit})
    result = bravesearch.run(query)
    return f"found {limit} news for {query}. {result}"




















@tool
def search_database(query: str, limit: int = 10) -> str:
    """
    Search the database for the given query and return the results.

    Args:
        query (str): The search query.
        limit (int, optional): The maximum number of results to return.
    """
    return f"found {limit} result for {query}."

