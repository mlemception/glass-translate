"""Screen capture: backends, backend selection and frame change detection."""
from .backends import DXCamCapture, MSSCapture
from .diff import ChangeDetector
from .factory import available_backends, create_capture

__all__ = [
    "ChangeDetector",
    "DXCamCapture",
    "MSSCapture",
    "available_backends",
    "create_capture",
]
