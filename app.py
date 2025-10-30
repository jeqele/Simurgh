"""
LangChain Calculator Agent using OpenRouter API
This script creates an agent that can perform mathematical calculations.

Prerequisites:
- pip install -r requirements.txt
- OPENROUTER_API_KEY environment variable set
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import os
import re

# Initialize ChatOpenAI with OpenRouter
# Ensure OPENROUTER_API_KEY environment variable is set
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY")  # Fallback to OPENAI_API_KEY if needed

llm = ChatOpenAI(
    model="google/gemma-2-27b-it",  # Using Gemma 2 27B Instruct model from OpenRouter
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)


# Create a calculator tool
def calculator(expression: str) -> str:
    """
    Evaluates mathematical expressions safely.
    Supports basic arithmetic operations: +, -, *, /, **, (), and common functions.
    """
    try:
        # Remove any text and keep only the mathematical expression
        expression = expression.strip()
        
        # Safety check - only allow mathematical characters
        if not re.match(r'^[0-9+\-*/().\s**]+$', expression):
            return f"Error: Invalid characters in expression. Only numbers and operators (+, -, *, /, **, ()) are allowed."
        
        # Evaluate the expression
        result = eval(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"

# Define the calculator tool
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for performing mathematical calculations. Input should be a valid mathematical expression like '2+2' or '10*5+3'. Supports +, -, *, /, ** (power), and parentheses."
    )
]

# Pull the prompt from LangChain hub or use a local one
try:
    prompt = hub.pull("hwchase17/react")
except:
    # Fallback to local prompt if hub is not accessible
    from langchain.prompts import PromptTemplate
    
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# Test the calculator agent
def main():
    print("=== LangChain Calculator Agent with OpenRouter API ===\n")
    
    # Check if API key is set
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Error: Either OPENROUTER_API_KEY or OPENAI_API_KEY environment variable must be set.")
        print("Please set your OpenRouter API key before running this script.")
        return

    test_questions = [
        "What is 25 + 37?",
        "Calculate 144 divided by 12",
        "What is 5 to the power of 3?",
        "Compute (10 + 5) * 3 - 8",
        "What is 100 / 4 + 50 * 2?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        try:
            response = agent_executor.invoke({"input": question})
            print(f"\n[SUCCESS] Final Answer: {response['output']}\n")
        except Exception as e:
            print(f"\n[ERROR] Error: {str(e)}\n")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Calculator Mode (type 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        user_input = input("Ask a calculation: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\n[SUCCESS] Answer: {response['output']}\n")
        except Exception as e:
            print(f"\n[ERROR] Error: {str(e)}\n")

if __name__ == "__main__":
    main()