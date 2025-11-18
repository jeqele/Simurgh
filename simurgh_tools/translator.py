"""
Translation Tool

Translates text to a requested language using either Ollama or OpenRouter.
Controlled by TRANSLATION_SERVICE environment variable ('ollama' or 'openrouter').
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default service and models
DEFAULT_TRANSLATION_SERVICE = os.environ.get("TRANSLATION_SERVICE", "ollama")  # 'ollama' or 'openrouter'
DEFAULT_OLLAMA_MODEL = "gemma3:12b"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


def _read_file_content(file_path: str) -> str:
    """
    Read content from a file.
    
    Args:
        file_path (str): Path to the file to read.
    
    Returns:
        str: File content, or error message if file cannot be read.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            return f"Error: File is empty: {file_path}"
        
        return content
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def _is_file_path(text: str) -> bool:
    """
    Check if the input string is likely a file path.
    
    Args:
        text (str): The input string to check.
    
    Returns:
        bool: True if it appears to be a file path, False otherwise.
    """
    # Check if it looks like a file path
    # Common indicators: contains path separators, has file extension, exists as file
    if os.path.sep in text or '/' in text:
        # Check if it exists as a file
        if os.path.isfile(text):
            return True
        # Check if it has a file extension
        path_obj = Path(text)
        if path_obj.suffix:  # Has extension
            return True
    
    return False


def _translate_with_ollama(text: str, target_language: str, model: str) -> str:
    """
    Translate using Ollama.
    
    Args:
        text (str): The text to be translated.
        target_language (str): The target language.
        model (str): The Ollama model to use.
    
    Returns:
        str: Translated text or error message.
    """
    try:
        import ollama
        
        # Create translation prompt
        prompt = f"Translate the following text to {target_language}. Only return the translation, no additional text or explanations:\n\n{text}"

        # Make API call to Ollama
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.3,  # Lower temperature for more consistent translations
            }
        )

        translated_text = response['message']['content'].strip()
        return f"Translated to {target_language}: {translated_text}"
    
    except ImportError:
        return "Error: Ollama package is not installed. Install it with: pip install ollama"
    except ConnectionError as e:
        return f"Error: Cannot connect to Ollama service. Make sure Ollama is running. You can start it or switch to OpenRouter by setting TRANSLATION_SERVICE=openrouter. Original error: {str(e)}"
    except Exception as e:
        error_msg = str(e)
        # Check for model not found errors
        if "404" in error_msg or "not found" in error_msg.lower():
            # Check if the model name looks like an OpenRouter model
            if "/" in model and ("openai" in model.lower() or "google" in model.lower() or "anthropic" in model.lower() or "meta" in model.lower()):
                return f"Error: Model '{model}' appears to be an OpenRouter model, not an Ollama model. For Ollama, use models like 'gemma3:12b', 'llama3', 'mistral', etc. You can either:\n1. Set TRANSLATE_MODEL_NAME to a valid Ollama model (e.g., 'gemma3:12b')\n2. Switch to OpenRouter by setting TRANSLATION_SERVICE=openrouter\nOriginal error: {error_msg}"
            else:
                return f"Error: Ollama model '{model}' not found. Make sure the model is pulled: 'ollama pull {model}'. Common Ollama models: gemma3:12b, llama3, mistral, etc. Original error: {error_msg}"
        # Check for connection errors
        if "502" in error_msg or "connection" in error_msg.lower():
            return f"Error: Ollama service is not available (502/connection error). Make sure Ollama is running. You can start it or switch to OpenRouter by setting TRANSLATION_SERVICE=openrouter in your environment. Original error: {error_msg}"
        return f"Error during Ollama translation: {error_msg}"


def _translate_with_openrouter(text: str, target_language: str, model: str) -> str:
    """
    Translate using OpenRouter.
    
    Args:
        text (str): The text to be translated.
        target_language (str): The target language.
        model (str): The OpenRouter model to use.
    
    Returns:
        str: Translated text or error message.
    """
    try:
        from openai import OpenAI
        
        # Get API key from environment variable
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            return "Error: OPENROUTER_API_KEY environment variable is not set. Please set it in a .env file or as an environment variable."

        # Initialize OpenAI client configured for OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Create translation prompt
        prompt = f"Translate the following text to {target_language}. Only return the translation, no additional text or explanations:\n\n{text}"

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent translations
        )

        translated_text = response.choices[0].message.content.strip()
        return f"Translated to {target_language}: {translated_text}"
    
    except ImportError:
        return "Error: OpenAI package is not installed. Install it with: pip install openai"
    except Exception as e:
        return f"Error during OpenRouter translation: {str(e)}"


def translate_text(text_or_file: str, target_language: str, model: str = None) -> str:
    """
    Translates the given text or file content into the target language using either Ollama or OpenRouter.
    The service is determined by the TRANSLATION_SERVICE environment variable.
    
    If the input is a file path, the file content will be read and translated.
    If the input is text, it will be translated directly.

    Args:
        text_or_file (str): The text to be translated or path to a file containing text.
        target_language (str): The language to translate the text into (e.g., 'French', 'Spanish', 'German').
        model (str, optional): The model to use for translation. If None, uses default based on service.

    Returns:
        str: The translated text, or error message if translation fails.
    """
    # Check if input is a file path
    text = text_or_file
    file_path = None
    
    if _is_file_path(text_or_file):
        file_path = text_or_file
        file_content = _read_file_content(file_path)
        
        # Check if file reading failed
        if file_content.startswith("Error:"):
            return file_content
        
        text = file_content
    
    # Determine which service to use (read dynamically each time)
    translation_service = os.environ.get("TRANSLATION_SERVICE", "ollama").lower()
    
    # Set default model if not specified
    if model is None:
        if translation_service == "ollama":
            model = os.environ.get("TRANSLATE_MODEL_NAME", DEFAULT_OLLAMA_MODEL)
        else:  # openrouter
            model = os.environ.get("TRANSLATE_MODEL_NAME", DEFAULT_OPENROUTER_MODEL)
    
    # Validate model name matches service (basic check)
    if translation_service == "ollama":
        # Ollama models typically don't have slashes (except for some like llama3:8b)
        # OpenRouter models have format like "openai/gpt-4o-mini" or "google/gemini-2.0-flash-exp:free"
        if "/" in model and ("openai" in model.lower() or "google" in model.lower() or "anthropic" in model.lower() or "meta" in model.lower()):
            # This looks like an OpenRouter model being used with Ollama
            return f"Error: Model '{model}' appears to be an OpenRouter model, but TRANSLATION_SERVICE is set to 'ollama'. For Ollama, use models like 'gemma3:12b', 'llama3', 'mistral', etc. You can either:\n1. Set TRANSLATE_MODEL_NAME to a valid Ollama model (e.g., 'gemma3:12b')\n2. Switch to OpenRouter by setting TRANSLATION_SERVICE=openrouter"
    
    # Route to appropriate translation service
    if translation_service == "ollama":
        result = _translate_with_ollama(text, target_language, model)
        # If Ollama fails with connection error, try OpenRouter as fallback if available
        if result.startswith("Error:") and ("connection" in result.lower() or "502" in result or "not available" in result.lower()):
            # Check if OpenRouter is configured
            if os.getenv("OPENROUTER_API_KEY"):
                result = f"{result}\n\nAttempting fallback to OpenRouter...\n"
                openrouter_result = _translate_with_openrouter(text, target_language, os.environ.get("TRANSLATE_MODEL_NAME", DEFAULT_OPENROUTER_MODEL))
                result += openrouter_result
            else:
                result += "\n\nTip: To use OpenRouter as fallback, set OPENROUTER_API_KEY in your environment."
    elif translation_service == "openrouter":
        result = _translate_with_openrouter(text, target_language, model)
    else:
        return f"Error: Invalid TRANSLATION_SERVICE value '{translation_service}'. Must be 'ollama' or 'openrouter'."
    
    # If we translated from a file, add file info to the result
    if file_path:
        result = f"Translated from file: {file_path}\n\n{result}"
    
    return result

