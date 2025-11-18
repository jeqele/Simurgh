"""
Text File Saver Tool

Saves text content to a file.
"""

import os
from pathlib import Path


def save_text_to_file(text: str, file_path: str) -> str:
    """
    Save text content to a file.

    Args:
        text (str): The text content to save
        file_path (str): Path to the output file (will create directory if needed)

    Returns:
        str: Success message with file path, or error message if save fails
    """
    try:
        # Create directory if it doesn't exist
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Write text to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Get absolute path for the message
        abs_path = os.path.abspath(file_path)
        return f"Successfully saved text to: {abs_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

