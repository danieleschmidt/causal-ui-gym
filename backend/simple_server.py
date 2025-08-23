#!/usr/bin/env python3
"""
Simple FastAPI server without heavy dependencies for Generation 2 robustness.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Basic HTTP server without heavy ML dependencies 
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs
    import socketserver
except ImportError:
    print("Error: Basic Python HTTP modules not available")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    status: str
    timestamp: float
    version: str
    uptime: float

@dataclass 
class ExperimentData:
    id: str
    name: str
    description: str
    created_at: float
    status: str

class CausalUIGymHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for Causal UI Gym API"""
    
    def __init__(self, *args, **kwargs):
        self.start_time = time.time()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == "/health":
                self.handle_health_check()
            elif path == "/":
                self.handle_root()
            elif path == "/api/status":
                self.handle_api_status()
            elif path.startswith("/api/experiments"):
                self.handle_experiments_get()
            else:
                self.send_404()
        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}")
            self.send_error_response(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b''
            
            if path == "/api/experiments":
                self.handle_experiments_post(post_data)
            elif path.startswith("/api/interventions"):
                self.handle_interventions_post(post_data)
            else:
                self.send_404()
        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}")
            self.send_error_response(500, f"Internal server error: {str(e)}")
    
    def handle_health_check(self):
        """Health check endpoint"""
        health = HealthStatus(
            status="healthy",
            timestamp=time.time(),
            version="0.1.0",
            uptime=time.time() - self.start_time
        )
        self.send_json_response(asdict(health))
    
    def handle_root(self):
        """Root endpoint with API information"""
        response = {
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
        self.send_json_response(response)
    
    def handle_api_status(self):
        """API status endpoint"""
        response = {
            "status": "operational",
            "experiments_count": len(self.get_experiments()),
            "uptime": time.time() - self.start_time,
            "engine_type": "SimpleEngine"
        }
        self.send_json_response(response)
    
    def handle_experiments_get(self):
        """Get experiments"""
        experiments = self.get_experiments()
        self.send_json_response({"experiments": experiments})
    
    def handle_experiments_post(self, post_data: bytes):
        """Create new experiment"""
        try:
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
            
            experiment = ExperimentData(
                id=f"exp_{int(time.time() * 1000)}",
                name=data.get("name", "Untitled Experiment"),
                description=data.get("description", ""),
                created_at=time.time(),
                status="created"
            )
            
            # Store experiment (in-memory for simplicity)
            experiments = self.get_experiments()
            experiments.append(asdict(experiment))
            self.save_experiments(experiments)
            
            self.send_json_response(asdict(experiment), status_code=201)
            
        except json.JSONDecodeError as e:
            self.send_error_response(400, f"Invalid JSON: {str(e)}")
    
    def handle_interventions_post(self, post_data: bytes):
        """Handle intervention requests"""
        try:
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
            
            # Simple intervention response
            response = {
                "intervention_id": f"int_{int(time.time() * 1000)}",
                "variable": data.get("variable", "unknown"),
                "value": data.get("value", 0),
                "result": {
                    "success": True,
                    "outcome": data.get("value", 0) * 1.2,  # Simple mock calculation
                    "timestamp": time.time()
                }
            }
            
            self.send_json_response(response)
            
        except json.JSONDecodeError as e:
            self.send_error_response(400, f"Invalid JSON: {str(e)}")
    
    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get stored experiments"""
        try:
            if os.path.exists("experiments.json"):
                with open("experiments.json", "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load experiments: {e}")
        return []
    
    def save_experiments(self, experiments: List[Dict[str, Any]]):
        """Save experiments to file"""
        try:
            with open("experiments.json", "w") as f:
                json.dump(experiments, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save experiments: {e}")
    
    def send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response"""
        response_data = json.dumps(data, indent=2)
        
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # CORS
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-length', str(len(response_data)))
        self.end_headers()
        self.wfile.write(response_data.encode())
    
    def send_error_response(self, status_code: int, message: str):
        """Send error response"""
        error_data = {
            "error": True,
            "message": message,
            "status_code": status_code,
            "timestamp": time.time()
        }
        self.send_json_response(error_data, status_code)
    
    def send_404(self):
        """Send 404 response"""
        self.send_error_response(404, "Endpoint not found")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Threading HTTP Server for better concurrency"""
    allow_reuse_address = True
    daemon_threads = True

def run_server(port: int = 8000):
    """Run the simple HTTP server"""
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, CausalUIGymHandler)
    
    logger.info(f"🚀 Causal UI Gym Simple Server starting on port {port}")
    logger.info(f"📊 Health check: http://localhost:{port}/health")
    logger.info(f"🔍 API status: http://localhost:{port}/api/status")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        httpd.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal UI Gym Simple Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    args = parser.parse_args()
    
    run_server(args.port)