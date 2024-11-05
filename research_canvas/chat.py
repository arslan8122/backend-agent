from typing import List, Dict, cast, Union
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain.tools import tool
from copilotkit.langchain import copilotkit_customize_config
from research_canvas.state import AgentState
from research_canvas.model import get_model
from research_canvas.download import get_resource
from research_canvas.state import BlogPost ,QuoteInfographic ,ComparisonInfographic ,StepsInfographic

@tool
def Search(queries: List[str]):
    """Search for references and inspiration for the blog post."""
    pass

@tool
def WriteBlogPost(blog_post: BlogPost):
    """Write or update the blog post with title and content."""
    pass

@tool
def GenerateInfographic(quote_info: QuoteInfographic):
    """Generate an quote based on the blog content."""
    pass

async def chat_node(state: AgentState, config: RunnableConfig):
    """
    Blog Generator Chat Node
    
    Handles the conversation flow for blog generation and infographic creation, including:
    - Managing blog post creation and updates
    - Generating content-based infographics
    - Tracking resources and logs
    - Coordinating with the AI model
    """
    # Configure state emission for blog post and infographics
    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[
            {
                "state_key": "blog_post",
                "tool": "WriteBlogPost",
                "tool_argument": "blog_post",
            },
            {
                "state_key": "quote_info",
                "tool": "GenerateInfographic",
                "tool_argument": "quote_info",
            }
        ]
    )

    # Initialize or get existing state
    state["resources"] = state.get("resources", [])
    state["blog_post"] = state.get("blog_post", {"title": "", "content": ""})
    state["quote_info"] = state.get("quote_info", {"type": "", "qoute": "","source":"","context":"" })
    state["infographics"] = state.get("infographics", [])
    state["logs"] = state.get("logs", [])
    state["messages"] = state.get("messages", [])

    # Process resources
    resources = []
    for resource in state["resources"]:
        content = get_resource(resource["url"])
        if content == "ERROR":
            continue
        resources.append({
            **resource,
            "content": content
        })

    # Invoke the model with tools
    response = await get_model(state).bind_tools(
        [
            Search,
            WriteBlogPost,
            GenerateInfographic,
        ],
    ).ainvoke([
        SystemMessage(
            content=f"""
            You are a professional blog content creator. Your role is to:
            1. Write engaging blog posts with well-structured content and compelling titles
            2. Create infographics to visualize key concepts when appropriate
            
            Current blog state:
            Title: {state["blog_post"].get("title", "")}
            Content: {state["blog_post"].get("content", "")}
            
            Current Quote: {state["quote_info"].get("quote","")}
            
            Available resources:
            {resources}
            """
        ),
        *state["messages"],
    ], config)

    ai_message = cast(AIMessage, response)
    updated_messages = list(state["messages"])
    updated_messages.append(ai_message)

    # Handle tool calls
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            tool_message = None

            if tool_call["name"] == "WriteBlogPost":
                print(tool_call["args"])
                state["blog_post"] = tool_call["args"]["blog_post"]
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Blog post updated."
                )
                state["logs"].append({
                    "message": f"Updated blog post: {tool_call['args']['blog_post']['title']}",
                    "done": True
                })
            
            elif tool_call["name"] == "GenerateInfographic":
                print(tool_call["args"])
                state["quote_info"] = tool_call["args"]["quote_info"]
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Quote has been written updated."
                )
                state["logs"].append({
                    "message": f"Quote for blog",
                    "done": True
                })

            if tool_message:
                updated_messages.append(tool_message)

    return {
        "blog_post": state["blog_post"],
        "quote_info": state["quote_info"],
        "logs": state["logs"],
        "messages": updated_messages
    }