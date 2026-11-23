"""Environment verification script for Indian Invoice & Bill Extraction project.

Checks Python version, presence and versions of required libraries, and GPU availability.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys
from typing import Dict, List, Tuple


REQUIRED_LIBRARIES: List[Tuple[str, str]] = [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("ultralytics", "ultralytics"),
    ("paddleocr", "paddleocr"),
    ("easyocr", "easyocr"),
    ("pytesseract", "pytesseract"),
    ("streamlit", "streamlit"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("pyyaml", "yaml"),
    ("pytest", "pytest"),
    ("jiwer", "jiwer"),
    ("matplotlib", "matplotlib"),
    ("openpyxl", "openpyxl"),
]


def check_python_version() -> str:
    """Return the active Python version and architecture."""
    version = platform.python_version()
    architecture = platform.architecture()[0]
    return f"{version} ({architecture})"


def check_library_version(dist_name: str, module_name: str) -> str:
    """Check version of a library using metadata or direct import fallback."""
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        pass

    # Fallback to direct import inspection if metadata lookup fails
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "installed (version unknown)")
    except ImportError:
        return "not installed"


def check_gpu() -> Dict[str, str]:
    """Check if GPU acceleration is available via PyTorch or OpenCV."""
    status: Dict[str, str] = {}

    # Check PyTorch CUDA
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
                status["PyTorch CUDA"] = f"Available ({device_count} device(s): {', '.join(device_names)})"
                status["CUDA Version"] = torch.version.cuda or "Unknown"
            else:
                status["PyTorch CUDA"] = "Not available (CPU only)"
        except Exception as exc:  # pragma: no cover
            status["PyTorch CUDA"] = f"Error checking PyTorch: {exc}"
    else:
        status["PyTorch CUDA"] = "PyTorch not installed"

    # Check OpenCV CUDA
    if importlib.util.find_spec("cv2") is not None:
        try:
            import cv2  # type: ignore

            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            status["OpenCV CUDA Devices"] = str(cuda_count)
        except Exception:
            status["OpenCV CUDA Devices"] = "OpenCV CUDA module not available"
    else:
        status["OpenCV CUDA Devices"] = "OpenCV not installed"

    return status


def main() -> None:
    """Run environment check and display formatted diagnostics."""
    print("=" * 65)
    print("  INVOICE VISION ENVIRONMENT DIAGNOSTICS")
    print("=" * 65)

    py_ver = check_python_version()
    print(f"Python Version: {py_ver}")
    print(f"Executable:     {sys.executable}")
    print("-" * 65)

    print("Required Libraries Status:")
    print(f"  {'Library':<20} | {'Status / Version'}")
    print("  " + "-" * 18 + "-+-" + "-" * 40)
    for dist_name, mod_name in REQUIRED_LIBRARIES:
        ver = check_library_version(dist_name, mod_name)
        print(f"  {dist_name:<20} | {ver}")

    print("-" * 65)
    print("Hardware / GPU Acceleration:")
    gpu_info = check_gpu()
    for key, val in gpu_info.items():
        print(f"  {key:<22}: {val}")
    print("=" * 65)


if __name__ == "__main__":
    main()
