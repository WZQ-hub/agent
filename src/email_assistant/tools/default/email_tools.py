from langchain_core.tools import tool
from typing import Literal
from pydantic import Basemodel


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email send to {to} with subject {subject} and content {content}"


@tool
def triage_email(category: Literal["ignore", "notify", "respond"]) -> str:
    """Triage an email into one of three categories: ignore, notify, respond."""
    return f"Classification Decision: {category}"


@tool
class Done(Basemodel):
    """email has beed sent"""
    done: bool

@tool
class Question(Basemodel):
    """Question to ask user."""
    content: str