from typing import Dict, Union, List, cast
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain.tools import tool
from copilotkit.langchain import copilotkit_customize_config
from research_canvas.state import AgentState
from research_canvas.model import get_model

@tool
def GenerateInfographic(infographics: Dict[str, Union[str, List[str]]]):
    """Generate a content-based infographic based on the blog content."""
    pass

async def infographics_node(state: AgentState, config: RunnableConfig):
    """
    Infographics Generator Node
    
    Handles the generation of infographics based on blog content:
    - Analyzes blog content for key points
    - Creates different types of infographics (quotes, steps, comparisons)
    - Manages infographics state
    """
    # Configure state emission for infographics
    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[{
            "state_key": "infographics",
            "tool": "GenerateInfographic",
            "tool_argument": "infographics",
        }]
    )

    # Initialize or get existing state
    state["infographics"] = state.get("infographics", [])
    state["logs"] = state.get("logs", [])
    state["messages"] = state.get("messages", [])

    # Get blog content for analysis
    blog_post = state.get("blog_post", {"title": "", "content": ""})
    
    # Invoke the model with infographic generation tool
    response = await get_model(state).bind_tools(
        [GenerateInfographic],
    ).ainvoke([
        SystemMessage(
            content=f"""
            You are an expert infographic designer. Your role is to analyze the blog content and create compelling infographics that enhance the message.
            
            Create infographics in these formats based on the content:
            
            For quotes:
            {{
                "type": "quote",
                "quote": "The quote text",
                "source": "Quote source",
                "context": "Context information"
            }}
            
            For steps:
            {{
                "type": "steps",
                "title": "Process title",
                "steps": ["step1", "step2", ...],
                "description": "Process description"
            }}
            
            For comparisons:
            {{
                "type": "comparison",
                "title": "Comparison title",
                "left_side": ["point1", "point2", ...],
                "right_side": ["point1", "point2", ...],
                "comparison_aspect": "What's being compared"
            }}

            Current blog content:
            Title: {blog_post["title"]}
            Content: {blog_post["content"]}
            
            Existing Infographics: {len(state["infographics"])}
            """
        ),
        *state["messages"],
    ], config)

    ai_message = cast(AIMessage, response)
    updated_messages = list(state["messages"])
    updated_messages.append(ai_message)

    # Handle infographic generation
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] == "GenerateInfographic":
                new_infographic = tool_call["args"]["infographics"]
                state["infographics"].append(new_infographic)
                
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=f"Generated {new_infographic['type']} infographic successfully."
                )
                updated_messages.append(tool_message)
                
                state["logs"].append({
                    "message": f"Generated {new_infographic['type']} infographic: {new_infographic.get('title', '')}",
                    "done": False
                })

    return {
        "infographics": state["infographics"],
        "logs": state["logs"],
        "messages": updated_messages
    }