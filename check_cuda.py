"""
CUDA Diagnostic Script
Checks if CUDA is properly configured for PyTorch and Whisper.
"""

import sys

print("=" * 60)
print("CUDA Diagnostic Check")
print("=" * 60)
print()

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA compiled version: {torch.version.cuda if torch.version.cuda else 'None (CPU-only build)'}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    CUDA capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("  ⚠ PyTorch does not have CUDA support")
        print()
        print("  This usually means PyTorch was installed without CUDA.")
        print("  You need to reinstall PyTorch with CUDA support.")
        print()
except ImportError:
    print("✗ PyTorch is not installed")
    sys.exit(1)

print()

# Check NVIDIA drivers
print("Checking NVIDIA drivers...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ nvidia-smi is available")
        # Extract CUDA version from nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'CUDA Version' in line:
                print(f"  {line.strip()}")
                break
    else:
        print("✗ nvidia-smi failed")
except FileNotFoundError:
    print("✗ nvidia-smi not found - NVIDIA drivers may not be installed")
except subprocess.TimeoutExpired:
    print("⚠ nvidia-smi timed out")
except Exception as e:
    print(f"⚠ Error checking nvidia-smi: {e}")

print()

# Recommendations
if not torch.cuda.is_available():
    print("=" * 60)
    print("RECOMMENDATION: Install CUDA-enabled PyTorch")
    print("=" * 60)
    print()
    print("To enable GPU acceleration, you need to reinstall PyTorch with CUDA support.")
    print()
    print("First, uninstall the current PyTorch:")
    print("  pip uninstall torch torchvision torchaudio")
    print()
    print("Then install CUDA-enabled PyTorch:")
    print("  # For CUDA 11.8:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("  # For CUDA 12.1:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("Check your CUDA version with: nvidia-smi")
    print()
else:
    print("=" * 60)
    print("✓ CUDA is properly configured!")
    print("=" * 60)

