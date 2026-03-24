from typing import Optional, List, Dict

from langchain_core.tools import BaseTool



def get_tools(tool_names: Optional[List[str]]=None, include_email=False) -> List[BaseTool]:
    """Get specified tools or all tools if tool_names is None"""

    # import tools
    from email_assistant.tools.default.email_tools import write_email, Done, Question
    from email_assistant.tools.default.calendar_tools import schedule_meeting, check_calendar_availability

    tools = {
        "write_email": write_email,
        "done": Done,
        "question": Question,
        "check_calendar_availability": check_calendar_availability,
        "schedule_meeting": schedule_meeting,
    }
    if include_email:
        from email_assistant.tools.gmail.gmail_tools import (
                fetch_emails_tool,
                send_email_tool,
                check_calendar_tool,
                schedule_meeting_tool
        )

        tools.update({
            "fetch_emails_tool": fetch_emails_tool,
            "send_email_tool": send_email_tool,
            "check_calendar_tool": check_calendar_tool,
            "schedule_meeting_tool": schedule_meeting_tool
        })

    if tool_names is None:
        return list(tools.values())

    return [tools[name] for name in tools if tools is not None]


def get_tools_by_name(tools: Optional[List[BaseTool]] = None) -> Dict[str, BaseTool]:
    """Get a dictionary of tools mapped by name."""
    if tools is None:
        tools = get_tools()

    return {tool.name: tool for tool in tools}


