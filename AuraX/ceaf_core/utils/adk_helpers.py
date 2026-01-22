# ceaf_core/utils/adk_helpers.py
"""ADK Helper utilities to handle Google ADK specific configurations"""

import os
import logging

logger = logging.getLogger(__name__)


def configure_adk_warnings():
    """Configure ADK to suppress certain warnings"""
    # Suppress Google ADK default value warnings
    os.environ['GOOGLE_ADK_SUPPRESS_DEFAULT_VALUE_WARNING'] = 'true'

    # You can add other ADK configurations here
    logger.info("ADK warnings configured")


def create_function_tool_safe(func):
    """
    Wrapper to create FunctionTool with proper handling of default parameters.
    This helps avoid the Google ADK warning about default values.
    """
    from google.adk.tools import FunctionTool

    # Remove default values from function signature if needed
    # This is a placeholder - implement based on your specific needs
    return FunctionTool(func=func)