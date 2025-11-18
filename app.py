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
from simurgh_tools.video_converter import convert_video_to_audio
from simurgh_tools.audio_to_text import convert_audio_to_text
from simurgh_tools.text_file_saver import save_text_to_file
from simurgh_tools.cuda_checker import check_cuda_status
from simurgh_tools.translator import translate_text

# Initialize ChatOpenAI with OpenRouter
# Ensure OPENROUTER_API_KEY environment variable is set
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY")  # Fallback to OPENAI_API_KEY if needed

core_model_name = os.environ.get("CORE_MODEL_NAME")
llm = ChatOpenAI(
    model=core_model_name,  # Using Gemma 2 27B Instruct model from OpenRouter
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


# Create a video to audio converter tool wrapper
def video_to_audio_converter(input_str: str) -> str:
    """
    Convert video file to MP3 audio.
    Input format: "video_path" or "video_path|output_path"
    """
    try:
        # Parse input - support both single path and path|output_path format
        parts = input_str.split('|')
        video_path = parts[0].strip()
        output_path = parts[1].strip() if len(parts) > 1 else None
        
        return convert_video_to_audio(video_path, output_path)
    except Exception as e:
        return f"Error: {str(e)}"


# Create an audio to text converter tool wrapper
def audio_to_text_converter(input_str: str) -> str:
    """
    Convert audio file to text using Whisper.
    Input format: "audio_path" or "audio_path|model_size" or "audio_path|model_size|language" or "audio_path|model_size|language|output_file"
    Model sizes: tiny, base, small, medium, large, turbo (default: turbo)
    Output file: If not specified, saves to same folder as input with .txt extension
    """
    try:
        # Parse input - support audio_path, audio_path|model_size, audio_path|model_size|language, or audio_path|model_size|language|output_file
        parts = input_str.split('|')
        audio_path = parts[0].strip()
        model_size = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "turbo"
        language = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        output_file = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
        
        # Validate model size
        valid_models = ["tiny", "base", "small", "medium", "large", "turbo"]
        if model_size not in valid_models:
            model_size = "turbo"
        
        return convert_audio_to_text(audio_path, model_size, language, output_file)
    except Exception as e:
        return f"Error: {str(e)}"


# Create a text file saver tool wrapper
def text_file_saver(input_str: str) -> str:
    """
    Save text content to a file.
    Input format: "text|file_path"
    Note: If text contains '|', use the last '|' as separator
    """
    try:
        # Parse input - format: "text|file_path"
        # Use rsplit to split from the right, so text can contain |
        parts = input_str.rsplit('|', 1)  # Split only on last |
        if len(parts) < 2:
            return "Error: Invalid format. Expected 'text|file_path'"
        
        text = parts[0].strip()
        file_path = parts[1].strip()
        
        return save_text_to_file(text, file_path)
    except Exception as e:
        return f"Error: {str(e)}"


# Create a CUDA checker tool wrapper
def cuda_checker(input_str: str = "") -> str:
    """
    Check CUDA status and configuration.
    No input required - just call the function.
    """
    try:
        return check_cuda_status()
    except Exception as e:
        return f"Error checking CUDA status: {str(e)}"


# Create a translator tool wrapper
def translator(input_str: str) -> str:
    """
    Translate text or file content to a target language.
    Input format: "text_or_file_path|target_language" or "text_or_file_path|target_language|model"
    Can accept either text content or a file path. If it's a file path, the file content will be read and translated.
    Target language can be any language name (e.g., 'French', 'Spanish', 'German', 'Japanese', 'Persian', 'Farsi')
    Model is optional (default: gemma3:12b via Ollama)
    """
    try:
        # Parse input - format: "text_or_file|target_language" or "text_or_file|target_language|model"
        # Use rsplit to handle text that might contain |
        parts = input_str.rsplit('|', 2)  # Split from right, max 2 splits
        
        if len(parts) < 2:
            return "Error: Invalid format. Expected 'text_or_file_path|target_language' or 'text_or_file_path|target_language|model'"
        
        text_or_file = parts[0].strip()
        target_language = parts[1].strip()
        model = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        
        return translate_text(text_or_file, target_language, model)
    except Exception as e:
        return f"Error: {str(e)}"


# Define the tools
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for performing mathematical calculations. Input should be a valid mathematical expression like '2+2' or '10*5+3'. Supports +, -, *, /, ** (power), and parentheses."
    ),
    Tool(
        name="VideoToAudioConverter",
        func=video_to_audio_converter,
        description="Useful for converting video files (MP4, etc.) to MP3 audio files. Input should be the path to the video file. Optionally, you can specify output path using format: 'video_path|output_path'. If output path is not provided, it will use the same name as the video file with .mp3 extension. Example: 'video.mp4' or 'video.mp4|audio.mp3'"
    ),
    Tool(
        name="AudioToTextConverter",
        func=audio_to_text_converter,
        description="Useful for converting audio files (MP3, WAV, etc.) to text using Whisper speech recognition. Automatically saves transcription to a file. Input format: 'audio_path' or 'audio_path|model_size' or 'audio_path|model_size|language' or 'audio_path|model_size|language|output_file'. Model sizes: tiny (fastest), base, small, medium, large, turbo (default). Language is optional (auto-detected if not specified). Output file is optional (defaults to same folder as input with .txt extension). Examples: 'audio.mp3', 'audio.mp3|turbo', 'audio.mp3|turbo|en', 'audio.mp3|turbo|en|output.txt'"
    ),
    Tool(
        name="TextFileSaver",
        func=text_file_saver,
        description="Useful for saving text content to a file. Input format: 'text|file_path'. The tool will create the directory if it doesn't exist. Example: 'Hello world|output.txt' or 'Long text content|/path/to/file.txt'"
    ),
    Tool(
        name="CUDAChecker",
        func=cuda_checker,
        description="Useful for checking CUDA (GPU) status and configuration for PyTorch. Checks if CUDA is available, shows GPU information, and provides recommendations if CUDA is not properly configured. No input required - just call the function. Useful when you need to verify GPU availability for audio transcription or other GPU-accelerated tasks."
    ),
    Tool(
        name="Translator",
        func=translator,
        description="Useful for translating text or file content from one language to another. Uses either Ollama (default) or OpenRouter based on TRANSLATION_SERVICE environment variable. Input format: 'text_or_file_path|target_language' or 'text_or_file_path|target_language|model'. Can accept either direct text content or a file path - if it's a file path, the file content will be read and translated. Target language can be any language name (e.g., 'French', 'Spanish', 'German', 'Japanese', 'Chinese', 'Arabic', 'Persian', 'Farsi'). Model is optional (default: gemma3:12b for Ollama, openai/gpt-4o-mini for OpenRouter). Examples: 'Hello, how are you?|French', 'file.txt|Persian', 'D:\\path\\to\\file.txt|Spanish|gemma3:12b'. Set TRANSLATION_SERVICE='ollama' or 'openrouter' in environment to choose service."
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
    
    # for question in test_questions:
    #     print(f"\n{'='*60}")
    #     print(f"Question: {question}")
    #     print('='*60)
        
    #     try:
    #         response = agent_executor.invoke({"input": question})
    #         print(f"\n[SUCCESS] Final Answer: {response['output']}\n")
    #     except Exception as e:
    #         print(f"\n[ERROR] Error: {str(e)}\n")
    
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