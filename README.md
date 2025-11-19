# Simurgh - Multi-Tool Agent

A versatile LangChain agent that automates tasks like calculations, video/audio processing, transcription, and translation.

## Features

- Mathematical calculations
- Video to audio conversion
- Audio transcription (Whisper with GPU support)
- Text translation (Ollama or OpenRouter)
- File operations
- CUDA/GPU diagnostics

## Installation

1. Clone and setup:
```bash
git clone <your-repo-url>
cd simurgh
python -m venv .venv
.venv\Scripts\activate  # Windows
# or source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

2. **Optional - GPU support** (for faster transcription):
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python check_cuda.py
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:
```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

## Usage

```bash
python app.py
```

Ask the agent to:
- "What is 25 + 37?"
- "Convert video.mp4 to audio"
- "Transcribe audio.mp3 to text"
- "Translate 'Hello' to French"
- "Translate file.txt to Persian"
- "Check my CUDA status"

## Tools

- **Calculator**: Math operations
- **VideoToAudioConverter**: Extract audio from video
- **AudioToTextConverter**: Whisper transcription (GPU optional)
- **Translator**: Multi-language translation
- **TextFileSaver**: Save text to files
- **CUDAChecker**: GPU diagnostics

## Requirements

- Python 3.8+
- OpenRouter or OpenAI API key
- NVIDIA GPU with CUDA (optional, for GPU acceleration)
- Ollama (optional, for local translation)

## License

APACHE
