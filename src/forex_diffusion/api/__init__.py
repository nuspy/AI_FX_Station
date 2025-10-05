"""Real-time inference API"""
from .inference_service import create_app, InferenceService

__all__ = ["create_app", "InferenceService"]
