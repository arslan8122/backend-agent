from typing import List, Dict, cast, Union
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain.tools import tool
from copilotkit.langchain import copilotkit_customize_config
from research_canvas.state import AgentState, BlogPost, QuoteInfographic, ComparisonInfographic, StepsInfographic, StatisticsGroup, BarGroup
from research_canvas.model import get_model
from research_canvas.download import get_resource

@tool
def Search(queries: List[str]):
    """Search for references and inspiration for the blog post."""
    pass

@tool
def WriteBlogPost(blog_post: BlogPost):
    """Write or update the blog post with title and content."""
    pass

@tool
def GenerateQuoteInfographic(quote_info: QuoteInfographic):
    """Generate a quote-based infographic from the blog content."""
    pass

@tool
def GenerateStepsInfographic(steps_info: StepsInfographic):
    """Generate a steps-based infographic from the blog content."""
    pass

@tool
def GenerateComparisonInfographic(comparison_info: ComparisonInfographic):
    """Generate a comparison-based infographic from the blog content."""
    pass

@tool
def GenerateStatisticsInfographic(stats_info: StatisticsGroup):
    """Generate a statistics-based infographic from the blog content."""
    pass

@tool
def GenerateBarChartInfographic(bars_info: BarGroup):
    """Generate a bar chart infographic from the blog content."""
    pass

async def chat_node(state: AgentState, config: RunnableConfig):
    """
    Blog Generator Chat Node
    
    Handles the conversation flow for blog generation and infographic creation, including:
    - Managing blog post creation and updates
    - Generating content-based infographics (quotes, steps, comparisons, statistics, and bar charts)
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
                "tool": "GenerateQuoteInfographic",
                "tool_argument": "quote_info",
            },
            {
                "state_key": "steps_info",
                "tool": "GenerateStepsInfographic",
                "tool_argument": "steps_info",
            },
            {
                "state_key": "comparison_info",
                "tool": "GenerateComparisonInfographic",
                "tool_argument": "comparison_info",
            },
            {
                "state_key": "stats_info",
                "tool": "GenerateStatisticsInfographic",
                "tool_argument": "stats_info",
            },
            {
                "state_key": "bars_info",
                "tool": "GenerateBarChartInfographic",
                "tool_argument": "bars_info",
            }
        ]
    )

    # Initialize or get existing state
    state["resources"] = state.get("resources", [])
    state["blog_post"] = state.get("blog_post", {"title": "", "content": ""})
    state["quote_info"] = state.get("quote_info", {"type": "quote", "quote": "", "source": "", "context": ""})
    state["steps_info"] = state.get("steps_info", {"type": "steps", "title": "", "steps": [], "description": ""})
    state["comparison_info"] = state.get("comparison_info", {
        "type": "comparison",
        "title": "",
        "left_side": [],
        "right_side": [],
        "left_title": "",
        "right_title": "",
        "comparison_aspect": "",
        "description": "",
        "conclusion": ""
    })
    state["stats_info"] = state.get("stats_info", {
        "title": "",
        "description": "",
        "stats": []
    })
    state["bars_info"] = state.get("bars_info", {
        "title": "",
        "stats": []
    })
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
            GenerateQuoteInfographic,
            GenerateStepsInfographic,
            GenerateComparisonInfographic,
            GenerateStatisticsInfographic,
            GenerateBarChartInfographic,
        ],
    ).ainvoke([
        SystemMessage(
            content=f"""
            You are an expert blog content creator specializing in creating engaging, informative content with visual elements. 
            Your task is to generate comprehensive blog posts with accompanying infographics.

            AVAILABLE TOOLS AND THEIR DATA FORMATS:

            1. WriteBlogPost
            Format:
            {{
             "blog_post": {{
                "title": "string",
                "content": "string (exactly 3000 words)"
            }}
            }}

            2. GenerateQuoteInfographic
            Format:
            {{
             "quote_info": {{
                "type": "quote",
                "quote": "string",
                "source": "string",
                "context": "string"
            }}
            }}

            3. GenerateStepsInfographic
            Format:
            {{
           "steps_info": {{
                "type": "steps",
                "title": "string",
                "steps": ["step1", "step2", "step3", ...],
                "description": "string"
            }}
            }}

            4. GenerateComparisonInfographic
            Format:
            {{
            "comparison_info": {{
                "type": "comparison",
                "title": "string",
                "left_side": ["point1", "point2", ...],
                "right_side": ["point1", "point2", ...],
                "left_title": "string",
                "right_title": "string",
                "comparison_aspect": "string",
                "description": "string",
                "conclusion": "string"
            }}
            }}

            5. GenerateStatisticsInfographic
            Format:
            {{
            "stats_info": {{
                "title": "string",
                "description": "string",
                "stats": [
                    {{"value": "string", "label": "string"}},
                    {{"value": "string", "label": "string"}},
                    ...
                ]
            }}
            }}

            6. GenerateBarChartInfographic
            Format:
            {{
            "bars_info": {{
                "title": "string",
                "stats": [
                    {{"value": "number (percentage between 0-100)", "label": "string"}},
                    {{"value": "number (percentage between 0-100)", "label": "string"}},
                    ...
                ]
            }}
            }}

            WORKFLOW REQUIREMENTS:
            1. First call WriteBlogPost to create the main content of exactly 3000 words
            2. Then call GenerateQuoteInfographic with an impactful quote FROM the blog content
            3. Call GenerateStepsInfographic to create a step-by-step guide based on the blog content
            4. Call GenerateComparisonInfographic to create a comparative analysis from the blog content
            5. Call GenerateStatisticsInfographic to highlight key statistics from the blog content
            6. Finally call GenerateBarChartInfographic to visualize percentage-based data from the blog content (all values must be between 0-100)

            STRICT GUIDELINES:
            - Blog post must be exactly 3000 words
            - Each infographic must match its specific format exactly
            - The quote infographic must contain a single quote
            - The steps infographic must contain an array of steps
            - The comparison infographic must have equal numbers of points on both sides
            - The statistics infographic must contain at least 3 statistics with values and labels
            - The bar chart infographic must contain at least 2 bars with percentage values (0-100) and labels
            - All bar chart values must be expressed as percentages out of 100
            - Do not mix up the tools or their data formats
            - Generate exactly one of each type

            Current State:
            Blog Title: {state["blog_post"].get("title", "")}
            Blog Content: {state["blog_post"].get("content", "")}
            Current Quote: {state["quote_info"].get("quote", "")}
            Current Steps: {state["steps_info"].get("steps", [])}
            Current Comparison: {state["comparison_info"].get("title", "")}
            Current Statistics: {state["stats_info"].get("stats", [])}
            Current Bar Chart: {state["bars_info"].get("stats", [])}

            Available Resources:
            {resources}

            REQUIRED TOOL CALLING SEQUENCE:
            1. Call WriteBlogPost exactly once to generate a 3000-word blog post
            2. Call GenerateQuoteInfographic exactly once
            3. Call GenerateStepsInfographic exactly once
            4. Call GenerateComparisonInfographic exactly once
            5. Call GenerateStatisticsInfographic exactly once
            6. Call GenerateBarChartInfographic exactly once with percentage values (0-100)

            Ensure each tool receives data in the exact format specified above.
            """
        ),
        *state["messages"],
    ], config)
    ai_message = cast(AIMessage, response)
    updated_messages = list(state["messages"])
    updated_messages.append(ai_message)
    print(ai_message.tool_calls)
    
    # Handle tool calls
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            tool_message = None

            if tool_call["name"] == "WriteBlogPost":
                state["blog_post"] = tool_call["args"]["blog_post"]
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Blog post updated."
                )
                state["logs"].append({
                    "message": f"Updated blog post: {tool_call['args']['blog_post']['title']}",
                    "done": True
                })
            
            elif "quote_info" in tool_call["args"]:
                state["quote_info"] = tool_call["args"]["quote_info"]
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Quote infographic has been updated."
                )
                state["logs"].append({
                    "message": "Generated quote infographic",
                    "done": True
                })
                
            elif "steps_info" in tool_call["args"]:
                state["steps_info"] = tool_call["args"]["steps_info"]
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Steps infographic has been updated."
                )
                state["logs"].append({
                    "message": f"Generated steps infographic: {tool_call['args']['steps_info']['title']}",
                    "done": True
                })
                
            elif "comparison_info" in tool_call["args"]:
                state["comparison_info"] = tool_call["args"]["comparison_info"]
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Comparison infographic has been updated."
                )
                state["logs"].append({
                    "message": f"Generated comparison infographic: {tool_call['args']['comparison_info']['title']}",
                    "done": True
                })

            elif "stats_info" in tool_call["args"]:
                state["stats_info"] = tool_call["args"]["stats_info"]
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Statistics infographic has been updated."
                )
                state["logs"].append({
                    "message": f"Generated statistics infographic: {tool_call['args']['stats_info']['title']}",
                    "done": True
                })

            elif "bars_info" in tool_call["args"]:
                state["bars_info"] = tool_call["args"]["bars_info"]
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Bar chart infographic has been updated."
                )
                state["logs"].append({
                    "message": f"Generated bar chart infographic: {tool_call['args']['bars_info']['title']}",
                    "done": True
                })

            if tool_message:
                updated_messages.append(tool_message)

    return {
        "blog_post": state["blog_post"],
        "quote_info": state["quote_info"],
        "steps_info": state["steps_info"],
        "comparison_info": state["comparison_info"],
        "stats_info": state["stats_info"],
        "bars_info": state["bars_info"],
        "logs": state["logs"],
        "messages": updated_messages
    }