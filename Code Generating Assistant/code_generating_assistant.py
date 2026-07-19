from typing import Annotated
from typing_extensions import TypedDict, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import gradio as gr

class CodingState(TypedDict):
    task: str
    messages: Annotated[list, add_messages]
    code: Optional[str]
    error: Optional[str]

def generate_code(state: CodingState) -> dict:
    task = state.get("task", "")
    messages = state.get("messages", [])
    code = state.get("code", "")
    error = state.get("error", "")

    system_message = """
        "You are an expert programmer. Output ONLY valid, raw executable code. "
        "Do NOT include any markdown code blocks, backticks (```), or introductory/explanatory text."
    """

    if not error:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "{task}")
        ])
        formatted_messages = prompt.format_messages(task=task)
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            (
                "user",
                "The previous code throws an error. Please fix it accordingly.\n\n"
                "Task: {task}\n\n"
                "Previous code:\n\n{code}\n\n"
                "Error: {error}\n\n"
                "Provide only the fixed code."
            )
        ])
        formatted_messages = prompt.format_messages(task=task, code=code, error=error)
    
    inputs = messages + formatted_messages
    response = llm_with_tools.invoke(inputs)

    return {
        "messages": [response],
        "code": response.content
    }

def check_code(state: CodingState) -> dict:
    code = state.get("code", "")
    try:
        exec(code)
        return {
            "error": ""
        }
    except Exception as e:
        return {"error": str(e)}

def should_fix_code(state: CodingState):
    if state["error"]:
        return "generate"
    return END

@tool
def web_search(topic: str) -> str:
    """
    Searches the web for the given topic
    Args:
        topic: The input topic to search for
    Returns:
        str: Search results
    """
    results = TavilySearchResults(max_results=5).invoke(topic)
    return str(results)

def get_code(task: str) -> str:
    result = coding_agent.invoke({
        "task": task,
        "messages": [],
        "code": "",
        "error": ""
    })
    return result["code"]

tools = [web_search]

llm = ChatOllama(model="qwen3", temperature=0)
llm_with_tools = llm.bind_tools(tools)

workflow = StateGraph(CodingState)

workflow.add_node("generate", generate_code)
workflow.add_node("check", check_code)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check")
workflow.add_conditional_edges(
    "check",
    should_fix_code,
    {
        "generate": "generate",
        END: END
    }
)

coding_agent = workflow.compile()

interface = gr.Interface(
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter your coding task here"
    ),
    outputs=gr.TextArea(),
    title="Code Generating Assistant",
    description="Enter your coding task, and I will write the code for you!",
    fn=get_code
)

interface.launch(debug=True)