from langchain_core.tools import tool
from datetime import datetime


@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_time: int, preferred_day: datetime, start_time: int
) -> str:
    """Schedule a calendar meeting"""
    date_str = preferred_day.strftime("%A, %B %d, %Y")
    return f"Meeting '{subject}' scheduled on {date_str} at {start_time} for {duration_time} minutes with {len(attendees)} attendees"



@tool
def check_calendar_availability(day: str) -> str:
    """Check the availability for a given day"""
    return f"Available times on {day}"