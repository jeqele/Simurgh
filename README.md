# Simurgh - Multi-Tool Agent for Task Automation

Simurgh is a versatile multi-tool agent designed to help people easily connect and automate various tasks. It serves as an intelligent interface that can interact with multiple tools and services to streamline workflows and automate repetitive processes.

## Features

- Multi-tool integration and orchestration
- Task automation across different platforms and services
- Intelligent decision making and workflow management
- Easy connection to various APIs and services
- Natural language interaction for task management
- Video and audio processing capabilities
- GPU acceleration support for AI tasks
- Multi-language translation support

## Prerequisites

- Python 3.8 or higher
- API keys for services you wish to connect (as needed)
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- Ollama installed and running (optional, for local translation)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd simurgh
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install CUDA-enabled PyTorch for GPU acceleration**
   
   By default, PyTorch installs with CPU-only support. For GPU acceleration (recommended for audio transcription), you need to install CUDA-enabled PyTorch:
   
   ```bash
   # First, uninstall CPU-only version
   pip uninstall torch torchvision torchaudio
   
   # Then install CUDA-enabled version (check your CUDA version with: nvidia-smi)
   # For CUDA 12.1:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
   Verify CUDA installation:
   ```bash
   python check_cuda.py
   ```

## Configuration

1. Create a `.env` file in the project root:
   ```bash
   # Copy from example if available, or create new
   ```

2. Add your API keys and configuration to the `.env` file:
   ```env
   # Required: OpenRouter API key (or OpenAI API key as fallback)
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   # or
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: Core model name for the agent (defaults to model specified in code)
   CORE_MODEL_NAME=google/gemma-2-27b-it
   
   # Optional: Translation service ('ollama' or 'openrouter', default: 'ollama')
   TRANSLATION_SERVICE=ollama
   
   # Optional: Translation model name
   # For Ollama: gemma3:12b, llama3, mistral, etc.
   # For OpenRouter: openai/gpt-4o-mini, google/gemini-2.0-flash-exp:free, etc.
   TRANSLATE_MODEL_NAME=gemma3:12b
   ```

## Usage

Run the Simurgh agent:

```bash
python app.py
```

The agent will start in interactive mode and be ready to handle various tasks based on your requests.

## Capabilities

Simurgh can help with:

### Mathematical Calculations
- Perform arithmetic operations (+, -, *, /, **)
- Evaluate complex mathematical expressions
- Support for parentheses and order of operations

### Video Processing
- Convert video files (MP4, etc.) to MP3 audio files
- Extract audio from video content

### Audio Transcription
- Convert audio files (MP3, WAV, etc.) to text using OpenAI Whisper
- Support for multiple Whisper model sizes (tiny, base, small, medium, large, turbo)
- Automatic language detection or specify language
- GPU acceleration support for faster transcription
- Automatically saves transcriptions to text files

### Text File Operations
- Save text content to files
- Automatic directory creation

### Translation
- Translate text or file content to any target language
- Support for both Ollama (local) and OpenRouter (cloud) translation services
- Can translate direct text or read and translate file contents
- Supports multiple languages (French, Spanish, German, Japanese, Persian, Farsi, etc.)

### System Diagnostics
- Check CUDA/GPU status and configuration
- Verify PyTorch CUDA support
- Get GPU information and recommendations

## Example Usage

Once the agent is running, you can ask it to:

- "What is 25 + 37?"
- "Convert video.mp4 to audio"
- "Transcribe audio.mp3 to text"
- "Translate 'Hello, how are you?' to French"
- "Translate file.txt to Persian"
- "Check my CUDA status"
- "Save this text to output.txt: [your text]"

## Tools Package

The `simurgh_tools` package contains modular utilities:

- **video_converter**: Convert video files to audio
- **audio_to_text**: Transcribe audio using Whisper with GPU support
- **text_file_saver**: Save text content to files
- **cuda_checker**: Diagnostic tool for CUDA/GPU status
- **translator**: Translate text using Ollama or OpenRouter

## How It Works

This project uses:
- **LangChain** for creating the agent framework
- **OpenRouter API** (or OpenAI) for the core LLM
- **OpenAI Whisper** for audio transcription
- **MoviePy** for video processing
- **Ollama** or **OpenRouter** for translation services
- **PyTorch** for GPU acceleration (optional)
- Multiple tool integrations for various services
- AI-powered decision making for task orchestration
- Secure connection management for external services

## GPU Acceleration

For best performance with audio transcription, GPU acceleration is recommended:

1. Ensure you have an NVIDIA GPU with CUDA support
2. Install CUDA-enabled PyTorch (see Installation section)
3. Run `python check_cuda.py` to verify your setup
4. The audio transcription tool will automatically use GPU if available

## Translation Services

Simurgh supports two translation services:

1. **Ollama** (default): Local translation using models like gemma3:12b
   - Requires Ollama installed and running
   - No API costs
   - Set `TRANSLATION_SERVICE=ollama` in `.env`

2. **OpenRouter**: Cloud-based translation
   - Requires `OPENROUTER_API_KEY`
   - Uses models like openai/gpt-4o-mini
   - Set `TRANSLATION_SERVICE=openrouter` in `.env`

## Security Note

Simurgh includes security measures to ensure safe connections and operations with external services. Always keep your API keys secure and never commit them to version control.

## License

APACHE
