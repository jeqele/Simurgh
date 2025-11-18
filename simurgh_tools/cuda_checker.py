"""
CUDA Diagnostic Tool

Checks if CUDA is properly configured for PyTorch and provides diagnostic information.
"""

import sys


def check_cuda_status() -> str:
    """
    Check CUDA status and configuration for PyTorch.
    
    Returns:
        str: Diagnostic information about CUDA availability and configuration
    """
    result_lines = []
    
    # Check PyTorch
    try:
        import torch
        result_lines.append(f"✓ PyTorch version: {torch.__version__}")
        cuda_compiled = torch.version.cuda if torch.version.cuda else 'None (CPU-only build)'
        result_lines.append(f"  CUDA compiled version: {cuda_compiled}")
        result_lines.append(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            result_lines.append(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                result_lines.append(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                capability = torch.cuda.get_device_capability(i)
                result_lines.append(f"    CUDA capability: {capability}")
        else:
            result_lines.append("  ⚠ PyTorch does not have CUDA support")
            result_lines.append("  This usually means PyTorch was installed without CUDA.")
            result_lines.append("  You need to reinstall PyTorch with CUDA support.")
    except ImportError:
        result_lines.append("✗ PyTorch is not installed")
        return "\n".join(result_lines)
    
    result_lines.append("")
    
    # Check NVIDIA drivers
    result_lines.append("Checking NVIDIA drivers...")
    try:
        import subprocess
        nvidia_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if nvidia_result.returncode == 0:
            result_lines.append("✓ nvidia-smi is available")
            # Extract CUDA version from nvidia-smi output
            lines = nvidia_result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    result_lines.append(f"  {line.strip()}")
                    break
        else:
            result_lines.append("✗ nvidia-smi failed")
    except FileNotFoundError:
        result_lines.append("✗ nvidia-smi not found - NVIDIA drivers may not be installed")
    except subprocess.TimeoutExpired:
        result_lines.append("⚠ nvidia-smi timed out")
    except Exception as e:
        result_lines.append(f"⚠ Error checking nvidia-smi: {e}")
    
    result_lines.append("")
    
    # Recommendations
    try:
        import torch
        if not torch.cuda.is_available():
            result_lines.append("=" * 60)
            result_lines.append("RECOMMENDATION: Install CUDA-enabled PyTorch")
            result_lines.append("=" * 60)
            result_lines.append("")
            result_lines.append("To enable GPU acceleration, you need to reinstall PyTorch with CUDA support.")
            result_lines.append("")
            result_lines.append("First, uninstall the current PyTorch:")
            result_lines.append("  pip uninstall torch torchvision torchaudio")
            result_lines.append("")
            result_lines.append("Then install CUDA-enabled PyTorch:")
            result_lines.append("  # For CUDA 11.8:")
            result_lines.append("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            result_lines.append("")
            result_lines.append("  # For CUDA 12.1:")
            result_lines.append("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            result_lines.append("")
            result_lines.append("Check your CUDA version with: nvidia-smi")
        else:
            result_lines.append("=" * 60)
            result_lines.append("✓ CUDA is properly configured!")
            result_lines.append("=" * 60)
    except ImportError:
        pass
    
    return "\n".join(result_lines)

