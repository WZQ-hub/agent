from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

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
def search_database(query: str, limit: int = 10) -> str:
    """
    Search the database for the given query and return the results.

    Args:
        query (str): The search query.
        limit (int, optional): The maximum number of results to return.
    """
    return f"found {limit} result for {query}."

