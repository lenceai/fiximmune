#!/usr/bin/env python3
"""
Script to launch the ML API service for FixImmune.

This script starts the FastAPI server for serving ML model predictions.
"""

import os
import argparse
import logging
from pathlib import Path

from api.model_api import start_model_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the FixImmune ML API service.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Port to bind the server to (default: 8001)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Start the API server
    logger.info(f"Starting ML API server on {args.host}:{args.port}")
    start_model_api(host=args.host, port=args.port)


if __name__ == "__main__":
    main() 