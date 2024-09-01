# Import necessary modules
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from IPython.display import Image, display

# Load environment variables from the .env file
load_dotenv()

# Retrieve API keys from environment variables
groq_key = os.getenv('GROQ_API_KEY')
langsmith_key = os.getenv('LANGSMITH_API_KEY')

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"

# Initialize the ChatGroq model with the API key
llm = ChatGroq(groq_api_key=groq_key, model_name="Gemma2-9b-It")

# Define a typed dictionary class to represent the state with annotated messages
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

# Create a state graph builder with the defined state structure
graph_builder = StateGraph(State)

# Define the chatbot function that interacts with the ChatGroq model
def chatbot(state: State):
    return {"messages": llm.invoke(state['messages'])}

# Add the chatbot node to the state graph
graph_builder.add_node("chatbot", chatbot)

# Display the current state of the graph builder
graph_builder

# Add edges to the graph, defining the flow from START to chatbot to END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph into a runnable state machine
graph = graph_builder.compile()

# Attempt to display the graph structure as an image
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Interactive chatbot loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Good Bye")
        break
    # Stream events through the graph with the user's input
    for event in graph.stream({'messages': ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"].content)
