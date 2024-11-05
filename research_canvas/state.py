"""
This is the state definition for the Blog Generator AI.
It defines the state of the agent and the state of the conversation.
"""

from typing import List, TypedDict, Dict , Literal
from langgraph.graph import MessagesState

class Resource(TypedDict):
    """
    Represents a resource. Give it a good title and a short description.
    """
    url: str
    title: str
    description: str

class Log(TypedDict):
    """
    Represents a log of an action performed by the agent.
    """
    message: str
    done: bool

class BlogPost(TypedDict):
    """
    Represents a blog post with title and content.
    """
    title: str
    content: str

class QuoteInfographic(TypedDict):
    """
    Represents a quote-based infographic for impactful statements.
    """
    type: Literal["quote"]
    quote: str
    source: str
    context: str

class StepsInfographic(TypedDict):
    """
    Represents a steps/flow-based infographic.
    """
    type: Literal["steps"]
    title: str
    steps: List[str]
    description: str

class ComparisonInfographic(TypedDict):
    """
    Represents a comparison-based infographic.
    """
    type: Literal["comparison"]
    title: str
    left_side: List[str]
    right_side: List[str]
    comparison_aspect: str

Infographic = QuoteInfographic | StepsInfographic | ComparisonInfographic

class AgentState(MessagesState):
    """
    State for the blog generation agent.
    Inherits from MessagesState for conversation management.
    """
    model: str
    infographics: List[Infographic]
    blog_post: BlogPost  # Current blog post being worked on
    quote_info:QuoteInfographic
    resources: List[Resource]  # Reference materials
    logs: List[Log] 