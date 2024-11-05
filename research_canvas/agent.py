# pylint: disable=line-too-long, unused-import
import json
from typing import cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from research_canvas.state import AgentState
from research_canvas.download import download_node
from research_canvas.chat import chat_node
from research_canvas.search import search_node
from research_canvas.infographics import infographics_node
from research_canvas.delete import delete_node, perform_delete_node

# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("search_node", search_node)
workflow.add_node("delete_node", delete_node)
workflow.add_node("perform_delete_node", perform_delete_node)

def route(state):
        """Route after the chat node."""
        messages = state.get("messages", [])
        if not messages:
            return END
            
        last_message = messages[-1]
        
        if isinstance(last_message, AIMessage):
            ai_message = cast(AIMessage, last_message)
            if not ai_message.tool_calls:
                return END
                
            tool_name = ai_message.tool_calls[0]["name"]
            if tool_name == "Search":
                return "search_node"
            elif tool_name == "DeleteResources":
                return "delete_node"
                
        if isinstance(last_message, ToolMessage):
            return "chat_node"

        return END

memory = MemorySaver()
# Change entry point to chat_node
workflow.set_entry_point("chat_node")
workflow.add_conditional_edges("chat_node", route, ["search_node", "chat_node", "delete_node", END])
workflow.add_edge("search_node", "chat_node")
workflow.add_edge("delete_node", "perform_delete_node")
workflow.add_edge("perform_delete_node", "chat_node")
graph = workflow.compile(checkpointer=memory, interrupt_after=["delete_node"])