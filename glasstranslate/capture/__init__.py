"""Screen capture: backends, backend selection and frame change detection."""
from .backends import DXCamCapture, MSSCapture
from .diff import ChangeDetector
from .factory import available_backends, create_capture
from .static import StaticPageCapture

__all__ = [
    "ChangeDetector",
    "DXCamCapture",
    "MSSCapture",
    "StaticPageCapture",
    "available_backends",
    "create_capture",
]
