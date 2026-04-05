import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END

from email_assistant.prompts import triage_user_prompt, triage_system_prompt, default_background, \
    default_triage_instructions, default_response_preferences, default_cal_preferences, agent_system_prompt_hitl
from email_assistant.schemas import RouterSchema, State, StateInput
from email_assistant.tools import get_tools, get_tools_by_name
from typing import Literal

from email_assistant.tools.default.prompt_templates import HITL_TOOLS_PROMPT
from utils import parse_email, format_email_markdown, format_for_display

load_dotenv(".env")
api_key = os.getenv("API_KEY")


tools = get_tools(["write_email", "schedule_meeting", "check_calendar_availability", "Question", "Done"])
tools_by_name = get_tools_by_name(tools)

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.1-TEE",
    base_url="https://llm.chutes.ai/v1",
    api_key=api_key,
    temperature=0.1,
)
llm_router = llm.with_structured_output(RouterSchema)


llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.1-TEE",
    base_url="https://llm.chutes.ai/v1",
    api_key=api_key,
    temperature=0.1,
)
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="required")



#Router
def triage_router(state: State) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """


    #parse email input
    author, to, subject, email_thread = parse_email(state["email_input"])
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    #Create email markdown for Agent Inbox in case of notification
    email_markdown = format_email_markdown(subject, author, to, email_thread)


    #format system prompt
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

    result = llm_router.invoke(
        [
            {
                "role": "system", "content": system_prompt
            },
            {
                "role": "user", "content": user_prompt
            }
        ]
    )

    classification = result.classification

    #Process the classification decision
    if classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")

        #next node
        goto = END

        #update state
        update = {
            "classification_decision": classification,
        }

    elif classification == "response":
        print("📧 Classification: RESPOND - This email requires a response")

        #next node
        goto = "response_agent"

        #update state
        update = {
            "classification_decision": classification,
            "message": [{
                "role": "user",
                "content": f"response to the email: {email_markdown}"
            }]
        }
    elif classification == "notice":
        print("🔔 Classification: NOTIFY - This email contains important information")

        #next node
        goto = "triage_interrupt_handler"

        #update
        update = {
            "classification_decision": classification,
        }
    else:
        raise ValueError(f"Invalid classification: {classification}")

    return Command(goto=goto, update=update)


def triage_interrupt_handler(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Handle interrupts from the triage step"""

    author, to, subject, email_thread = parse_email(state["email_input"])
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    #create message
    messages = [{
        "role": "user",
        "content": f"interrupted by the user: {email_markdown}"
    }]

    # Create interrupt for Agent Inbox
    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {}
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        # Email to show in Agent Inbox
        "description": email_markdown,
    }

    # Agent Inbox responds with a list
    response = interrupt([request])[0]

    # If user provides feedback, go to response agent and use feedback to respond to email
    if response["type"] == "response":
        user_input = response["args"]
        messages.append({
            "role": "user",
            "content": f"User want to reply to the email. Use this feedback to respond: {user_input}"
        })

        goto = "response_agent"

    elif response["type"] == "ignore":
      goto = END

    else:
        raise ValueError(f"Invalid response: {response}")

    update = {
        "message": messages
    }
    return Command(goto=goto, update=update)

def llm_call(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt_hitl.format(
                        tools_prompt=HITL_TOOLS_PROMPT,
                        background=default_background,
                        response_preferences=default_response_preferences,
                        cal_preferences=default_cal_preferences
                    )}
                ]
                + state["messages"]
            )
        ]
    }

def interrupt_handler(state: State) -> Command[Literal["llm_call", "__end__"]]:
    """Creates an interrupt for human review of tool calls"""

    result = []
    goto = "llm_call"

    for tool_call in state["messages"][-1].tool_calls:
        hitl_tools = ["write_email", "schedule_meeting", "Question"]
        if tool_call["name"] not in hitl_tools:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({
                "role": "tool",
                "content": observation,
                "tool_call_id": tool_call["id"]
            })
            continue

        email_input = state["email_input"]
        author, to, subject, email_thread = parse_email(email_input)
        original_email_markdown = format_email_markdown(subject, author, to, email_thread)

        tool_display = format_for_display(tool_call)
        description = original_email_markdown + tool_display


        if tool_call["name"] == "write_email":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "schedule_meeting":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "Question":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            }
        else:
            raise ValueError(f"Invalid tool: {tool_call['name']}")

        request = {
            "action_request": {
                "action": tool_call["name"],
                "args": tool_call["args"]
            },
            "config": config,
            "description": description,
        }

        response = interrupt([request])[0]

        if response["type"] == "accept":
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_call["id"],
                }
            )
        elif response["type"] == "edit":
            tool = tools_by_name[tool_call["name"]]
            edited_args = response["args"]["args"]
            ai_message = state["messages"][-1]
            current_id = tool_call["id"]

            updated_tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [
                {"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}
            ]

            result.append(ai_message.model_copy(update={"tool_calls": updated_tool_calls}))

            # Update the write_email tool call with the edited content from Agent Inbox
            if tool_call["name"] == "write_email":

                # Execute the tool with edited args
                observation = tool.invoke(edited_args)

                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

            # Update the schedule_meeting tool call with the edited content from Agent Inbox
            elif tool_call["name"] == "schedule_meeting":

                # Execute the tool with edited args
                observation = tool.invoke(edited_args)

                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

            # Catch all other tool calls
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":
            if tool_call["name"] == "write_email":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool",
                               "content": "User ignored this email draft. Ignore this email and end the workflow.",
                               "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            elif tool_call["name"] == "schedule_meeting":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool",
                               "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.",
                               "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            elif tool_call["name"] == "Question":
                # Don't execute the tool, and tell the agent how to proceed
                result.append(
                    {"role": "tool", "content": "User ignored this question. Ignore this email and end the workflow.",
                     "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            if tool_call["name"] == "write_email":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool",
                               "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}",
                               "tool_call_id": tool_call["id"]})
            elif tool_call["name"] == "schedule_meeting":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool",
                               "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}",
                               "tool_call_id": tool_call["id"]})
            elif tool_call["name"] == "Question":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool",
                               "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}",
                               "tool_call_id": tool_call["id"]})
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

            # Catch all other responses
        else:
            raise ValueError(f"Invalid response: {response}")

            # Update the state
        update = {
            "messages": result,
        }

        return Command(goto=goto, update=update)


# Conditional edge function
def should_continue(state: State) -> Literal["interrupt_handler", "__end__"]:
    """Route to tool handler, or end inf Done tool called"""
    message = state["messages"]
    last_message = message[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
            else:
                return "interrupt_handler"


agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)


agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        END: END,
    },
)
response_agent = agent_builder.compile()

overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")

)
email_assistant = overall_workflow.compile()










