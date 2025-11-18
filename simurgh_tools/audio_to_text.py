"""
Audio to Text Converter Tool

Converts audio files to text using OpenAI's Whisper model (local).
Uses GPU (CUDA) if available, otherwise falls back to CPU.
"""

import os
import whisper
import torch
from pathlib import Path


# Cache models by size to avoid reloading them every time
_model_cache = {}
_device = None


def _get_device():
    """
    Get the appropriate device (GPU if CUDA is available, otherwise CPU).
    
    Returns:
        str: Device name ('cuda' or 'cpu')
    """
    global _device
    
    if _device is None:
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            _device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✓ Using GPU: {gpu_name} (CUDA {cuda_version})")
        else:
            _device = "cpu"
            # Provide diagnostic information
            print("⚠ CUDA not available, using CPU")
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA compiled version: {torch.version.cuda if torch.version.cuda else 'None (CPU-only build)'}")
            
            # Check if CUDA is installed on system but PyTorch doesn't support it
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    print("  ⚠ NVIDIA GPU detected but PyTorch doesn't have CUDA support!")
                    print("  💡 Install CUDA-enabled PyTorch:")
                    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            except:
                pass  # nvidia-smi not available or error
    
    return _device


def _load_whisper_model(model_size: str = "turbo"):
    """
    Load Whisper model with caching (cached by model size).
    Uses GPU if CUDA is available, otherwise uses CPU.
    
    Args:
        model_size (str): Size of the Whisper model. Options: tiny, base, small, medium, large, turbo
                         Default: turbo (good balance of speed and accuracy)
    
    Returns:
        whisper.Model: Loaded Whisper model
    """
    global _model_cache
    
    device = _get_device()
    cache_key = f"{model_size}_{device}"
    
    if cache_key not in _model_cache:
        try:
            _model_cache[cache_key] = whisper.load_model(model_size, device=device)
        except Exception as e:
            raise Exception(f"Failed to load Whisper model '{model_size}' on {device}: {str(e)}")
    
    return _model_cache[cache_key]


def convert_audio_to_text(audio_path: str, model_size: str = "turbo", language: str = None, output_file: str = None) -> str:
    """
    Convert an audio file to text using Whisper.

    Args:
        audio_path (str): Path to the input audio file (MP3, WAV, etc.)
        model_size (str, optional): Size of the Whisper model. Options: tiny, base, small, medium, large, turbo.
                                   Default: turbo. Larger models are more accurate but slower.
        language (str, optional): Language code (e.g., 'en', 'es', 'fr'). If None, auto-detects.
        output_file (str, optional): Path to save the transcription. If None, uses same folder as input with .txt extension.

    Returns:
        str: Transcribed text and file location (if saved), or error message if transcription fails
    """
    # Validate input file
    if not os.path.exists(audio_path):
        return f"Error: Audio file not found: {audio_path}"

    try:
        # Load the Whisper model
        model = _load_whisper_model(model_size)
        
        # Transcribe the audio
        result = model.transcribe(
            audio_path,
            language=language,
            verbose=False
        )
        
        # Extract the transcribed text
        transcribed_text = result["text"].strip()
        
        # Always save to file (default: same folder as input with .txt extension)
        # Generate output file path if not provided
        if output_file is None:
            audio_file = Path(audio_path)
            output_file = str(audio_file.with_suffix('.txt'))
        
        # Create directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write text to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcribed_text)
        
        saved_file_path = os.path.abspath(output_file)
        
        # Build response message
        response = f"Transcription: {transcribed_text}"
        if saved_file_path:
            response += f"\n\nSaved to file: {saved_file_path}"
        
        return response
    except Exception as e:
        return f"Error during transcription: {str(e)}"

