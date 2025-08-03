"""
FastAPI server for Causal UI Gym backend.

This module sets up the main FastAPI application with all necessary
middleware, routing, and configuration for the causal inference API.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
import os

from .routes import experiments, interventions, agents, metrics
from ..engine.causal_engine import JaxCausalEngine


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state for the application
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Causal UI Gym backend...")
    
    # Initialize JAX causal engine
    app_state["causal_engine"] = JaxCausalEngine(random_seed=42)
    logger.info("JAX causal engine initialized")
    
    # Initialize experiment storage (in production, use proper database)
    app_state["experiments"] = {}
    app_state["results"] = {}
    app_state["agents"] = {}
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Causal UI Gym backend...")
    app_state.clear()


# Create FastAPI application
app = FastAPI(
    title="Causal UI Gym API",
    description="Backend API for testing LLM causal reasoning through interactive UI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"{request.method} {request.url.path} - Start")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "engine_initialized": "causal_engine" in app_state,
        "version": "0.1.0"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Causal UI Gym API",
        "description": "Backend API for testing LLM causal reasoning",
        "version": "0.1.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "experiments": "/api/experiments",
            "interventions": "/api/interventions", 
            "agents": "/api/agents",
            "metrics": "/api/metrics"
        }
    }


# API status endpoint
@app.get("/api/status")
async def api_status():
    """Get API status and statistics."""
    return {
        "status": "operational",
        "experiments_count": len(app_state.get("experiments", {})),
        "results_count": len(app_state.get("results", {})),
        "agents_count": len(app_state.get("agents", {})),
        "engine_type": type(app_state.get("causal_engine")).__name__
    }


# Include routers
app.include_router(experiments.router, prefix="/api", tags=["experiments"])
app.include_router(interventions.router, prefix="/api", tags=["interventions"])
app.include_router(agents.router, prefix="/api", tags=["agents"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])


# Dependency to get causal engine
def get_causal_engine() -> JaxCausalEngine:
    """Dependency to get the causal engine instance."""
    engine = app_state.get("causal_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="Causal engine not initialized")
    return engine


# Dependency to get application state
def get_app_state() -> Dict[str, Any]:
    """Dependency to get application state."""
    return app_state


# Export dependencies for use in route modules
__all__ = ["app", "get_causal_engine", "get_app_state"]