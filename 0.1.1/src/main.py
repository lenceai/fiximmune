#!/usr/bin/env python3
"""
FixImmune Main Application
--------------------------
Entry point for the FixImmune application that provides personalized
treatment recommendations for autoimmune conditions.
"""

import os
import logging
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .database.init_db import initialize_database, get_db
from .utils.sample_data import initialize_sample_data

# Load environment variables
load_dotenv()

# Configure logging
logging_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=os.getenv("APP_NAME", "FixImmune"),
    description="Personalized treatment recommendations for autoimmune conditions",
    version=os.getenv("APP_VERSION", "0.1.1"),
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    logger.info("Initializing database...")
    initialize_database()
    
    # Initialize sample data if in debug mode
    if os.getenv("DEBUG", "False").lower() == "true":
        logger.info("Debug mode enabled, initializing sample data...")
        db = next(get_db())
        try:
            initialize_sample_data(db)
        except Exception as e:
            logger.error(f"Error initializing sample data: {e}")
        finally:
            db.close()
    
    logger.info("Application startup complete.")

@app.get("/")
async def root():
    """Root endpoint that provides basic application information."""
    return {
        "application": os.getenv("APP_NAME", "FixImmune"),
        "version": os.getenv("APP_VERSION", "0.1.1"),
        "status": "active",
        "documentation": "/docs",
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "False").lower() == "true",
    ) 