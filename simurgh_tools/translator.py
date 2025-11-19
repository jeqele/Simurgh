"""
Translation Tool

Translates text to a requested language using either Ollama or OpenRouter.
Controlled by TRANSLATION_SERVICE environment variable ('ollama' or 'openrouter').
"""

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Helper functions to check if services are enabled
def _is_ollama_enabled() -> bool:
    """Check if Ollama is enabled based on .env configuration."""
    ollama_enabled = os.environ.get("OLLAMA_ENABLED_FOR_TRANSLATION", "").lower()
    return ollama_enabled == "true"

def _is_openrouter_enabled() -> bool:
    """Check if OpenRouter is enabled based on .env configuration."""
    openRouter_enabled = os.environ.get("OPENROUTER_ENABLED_FOR_TRANSLATION", "").lower()
    return openRouter_enabled == "true"

def _get_ollama_model() -> str:
    """Get Ollama model from .env, raise error if not set."""
    model = os.environ.get("TRANSLATE_MODEL_NAME_OLLAMA") or os.environ.get("OLLAMA_MODEL")
    if not model:
        raise ValueError("OLLAMA_MODEL or TRANSLATE_MODEL_NAME_OLLAMA must be set in .env file")
    return model

def _get_openrouter_model() -> str:
    """Get OpenRouter model from .env, raise error if not set."""
    model = os.environ.get("TRANSLATE_MODEL_NAME_OPENROUTER") or os.environ.get("OPENROUTER_MODEL")
    if not model:
        raise ValueError("OPENROUTER_MODEL or TRANSLATE_MODEL_NAME_OPENROUTER must be set in .env file")
    return model


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


def _extract_translated_text(translation_result: str) -> str:
    """
    Extract the actual translated text from the translation result string.
    
    Args:
        translation_result (str): The full translation result (may include "Translated to {language}:" prefix).
    
    Returns:
        str: The extracted translated text, or the original string if it's an error.
    """
    if translation_result.startswith("Error:"):
        return translation_result
    
    # Remove "Translated to {language}:" prefix if present
    if ":" in translation_result:
        parts = translation_result.split(":", 1)
        if len(parts) == 2 and "Translated to" in parts[0]:
            return parts[1].strip()
    
    return translation_result


def _translate_single_sentence(sentence: str, target_language: str, model: str = None) -> str:
    """
    Translate a single sentence using the appropriate service with fallback logic.
    
    Logic:
    - If Ollama is enabled: try Ollama first, if it fails and OpenRouter is enabled, try OpenRouter
    - If only OpenRouter is enabled: use OpenRouter
    
    Args:
        sentence (str): The sentence to translate.
        target_language (str): The target language.
        model (str, optional): The model to use. If None, uses model from .env based on service.
    
    Returns:
        str: The translated sentence, or error message.
    """
    ollama_enabled = _is_ollama_enabled()
    openrouter_enabled = _is_openrouter_enabled()
    
    # Determine which model to use
    if model is None:
        if ollama_enabled:
            try:
                model = _get_ollama_model()
            except ValueError as e:
                if openrouter_enabled:
                    # Fallback to OpenRouter if Ollama model not configured
                    model = _get_openrouter_model()
                    ollama_enabled = False
                else:
                    return f"Error: {str(e)}"
        elif openrouter_enabled:
            model = _get_openrouter_model()
        else:
            return "Error: Neither Ollama nor OpenRouter is enabled. Please configure at least one service in .env"
    
    # Try Ollama first if enabled
    if ollama_enabled:
        result = _translate_with_ollama(sentence, target_language, model)
        # If Ollama fails, try OpenRouter as fallback if available
        if result.startswith("Error:"):
            if openrouter_enabled:
                # Get OpenRouter model for fallback
                try:
                    openrouter_model = _get_openrouter_model()
                    result = _translate_with_openrouter(sentence, target_language, openrouter_model)
                except ValueError:
                    # If OpenRouter model not configured, return Ollama error
                    pass
        return result
    
    # If only OpenRouter is enabled, use it
    elif openrouter_enabled:
        result = _translate_with_openrouter(sentence, target_language, model)
        return result
    
    # Neither service is enabled
    return "Error: Neither Ollama nor OpenRouter is enabled. Please configure at least one service in .env"


def translate_text(text_or_file: str, target_language: str, model: str = None) -> str:
    """
    Translates the given text or file content into the target language using either Ollama or OpenRouter.
    The service is determined by the TRANSLATION_SERVICE environment variable.
    
    If the input is a file path, the file content will be read and translated.
    If the input is text, it will be translated directly.
    
    The text is split by "." into sentences, and each sentence is translated individually.
    Each translated sentence is printed at runtime, and the final result is saved to a file.

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
    
    # Determine which service to use and get model from .env
    ollama_enabled = _is_ollama_enabled()
    openrouter_enabled = _is_openrouter_enabled()
    
    # Set model from .env if not specified
    if model is None:
        if ollama_enabled:
            try:
                model = _get_ollama_model()
                print(f"Using Ollama model: {model}")
            except ValueError as e:
                if openrouter_enabled:
                    model = _get_openrouter_model()
                    print(f"Ollama model not configured, using OpenRouter model: {model}")
                    ollama_enabled = False
                else:
                    return f"Error: {str(e)}"
        elif openrouter_enabled:
            model = _get_openrouter_model()
            print(f"Using OpenRouter model: {model}")
        else:
            return "Error: Neither Ollama nor OpenRouter is enabled. Please configure at least one service in .env"
    
    # Split text by "." to get sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    if not sentences:
        return "Error: No sentences found in the text to translate."
    
    # Translate each sentence individually
    translated_sentences = []
    print(f"\nTranslating {len(sentences)} sentence(s)...\n")
    
    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] Translating: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
        
        # Translate the sentence (fallback logic handled inside)
        translation_result = _translate_single_sentence(sentence, target_language, model)
        
        # Check for errors
        if translation_result.startswith("Error:"):
            print(f"Error translating sentence {i}: {translation_result}")
            translated_sentences.append(sentence)  # Keep original on error
        else:
            # Extract the actual translated text
            translated_text = _extract_translated_text(translation_result)
            translated_sentences.append(translated_text)
            print(f"Translated: {translated_text}\n")
    
    # Join all translated sentences with "."
    final_translation = ". ".join(translated_sentences)
    
    # Add period at the end if the original text ended with one
    if text.rstrip().endswith("."):
        final_translation += "."
    
    # Save to file
    if file_path:
        # Generate output filename based on input file
        input_path = Path(file_path)
        output_filename = f"{input_path.stem}_translated_{target_language.lower().replace(' ', '_')}{input_path.suffix}"
        output_path = input_path.parent / output_filename
    else:
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"translated_{target_language.lower().replace(' ', '_')}_{timestamp}.txt"
        output_path = Path(output_filename)
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_translation)
        
        abs_path = os.path.abspath(output_path)
        print(f"\nTranslation saved to: {abs_path}\n")
        
        result = f"Translated to {target_language}:\n\n{final_translation}\n\nSaved to: {abs_path}"
        
        if file_path:
            result = f"Translated from file: {file_path}\n\n{result}"
        
        return result
    except Exception as e:
        error_msg = f"Error saving translation to file: {str(e)}"
        print(f"\n{error_msg}\n")
        result = f"Translated to {target_language}:\n\n{final_translation}\n\n{error_msg}"
        if file_path:
            result = f"Translated from file: {file_path}\n\n{result}"
        return result

