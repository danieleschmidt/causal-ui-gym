# API Documentation Framework

This document outlines the API documentation strategy for the Causal UI Gym project.

## Overview

The project uses FastAPI's automatic OpenAPI generation combined with custom documentation enhancements for comprehensive API documentation.

## OpenAPI Configuration

### FastAPI Setup

The main FastAPI application should include comprehensive metadata:

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="Causal UI Gym API",
    description="""
    A comprehensive platform for causal inference experimentation and learning.
    
    ## Features
    
    * **Causal Graph Analysis**: Interactive causal graph construction and analysis
    * **Experiment Management**: Design and run causal inference experiments
    * **ML Integration**: JAX-powered causal reasoning computations
    * **Data Visualization**: D3.js-based interactive visualizations
    
    ## Authentication
    
    The API uses JWT tokens for authentication. Include the token in the Authorization header:
    
    ```
    Authorization: Bearer <your_jwt_token>
    ```
    """,
    version="1.0.0",
    contact={
        "name": "Causal UI Gym Team",
        "url": "https://github.com/your-org/causal-ui-gym",
        "email": "support@causal-ui-gym.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "https://api.causal-ui-gym.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.causal-ui-gym.com", 
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://causal-ui-gym.com/logo.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

### Response Models

All API endpoints should use Pydantic models for request/response validation:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class CausalGraphRequest(BaseModel):
    """Request model for creating a causal graph"""
    nodes: List[str] = Field(..., description="List of variable names")
    edges: List[tuple] = Field(..., description="List of (source, target) pairs")
    metadata: Optional[dict] = Field(None, description="Additional graph metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "nodes": ["X", "Y", "Z"],
                "edges": [("X", "Y"), ("Y", "Z")],
                "metadata": {"experiment_id": "exp_001"}
            }
        }

class CausalGraphResponse(BaseModel):
    """Response model for causal graph operations"""
    graph_id: str = Field(..., description="Unique graph identifier")
    nodes: List[str]
    edges: List[tuple]
    created_at: datetime
    analysis_results: Optional[dict] = None
    
    class Config:
        schema_extra = {
            "example": {
                "graph_id": "graph_123",
                "nodes": ["X", "Y", "Z"],
                "edges": [("X", "Y"), ("Y", "Z")],
                "created_at": "2025-01-31T10:00:00Z",
                "analysis_results": {
                    "d_separation": True,
                    "confounders": ["Z"]
                }
            }
        }
```

### Endpoint Documentation

Use comprehensive docstrings and response models:

```python
@app.post("/api/v1/graphs", response_model=CausalGraphResponse, tags=["Causal Graphs"])
async def create_causal_graph(
    graph: CausalGraphRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new causal graph for analysis.
    
    This endpoint creates a new causal graph from the provided nodes and edges,
    validates the graph structure, and returns the graph ID for further operations.
    
    - **nodes**: List of variable names (must be unique)
    - **edges**: List of directed edges as (source, target) pairs
    - **metadata**: Optional metadata for the graph
    
    Returns the created graph with analysis results if requested.
    
    Raises:
        - **400**: Invalid graph structure or duplicate nodes
        - **401**: Authentication required
        - **422**: Validation error in request data
    """
    # Implementation here
    pass
```

### Error Handling

Standardized error responses:

```python
from fastapi import HTTPException
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "error_code": "INVALID_GRAPH_STRUCTURE",
                "message": "The provided graph contains cycles",
                "details": {"cycle": ["X", "Y", "X"]},
                "timestamp": "2025-01-31T10:00:00Z"
            }
        }

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message=str(exc)
        ).dict()
    )
```

## Documentation Deployment

### Local Development

```bash
# Start FastAPI with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
# OpenAPI JSON: http://localhost:8000/openapi.json
```

### Production Setup

```dockerfile
# Add to Dockerfile
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Documentation Export

Generate static documentation:

```python
# scripts/generate_docs.py
import json
from main import app

def export_openapi_spec():
    """Export OpenAPI specification to JSON file"""
    openapi_spec = app.openapi()
    with open("docs/openapi.json", "w") as f:
        json.dump(openapi_spec, f, indent=2)
    print("OpenAPI specification exported to docs/openapi.json")

if __name__ == "__main__":
    export_openapi_spec()
```

## Integration with Frontend

### TypeScript Client Generation

```bash
# Install OpenAPI TypeScript generator
npm install --save-dev @openapitools/openapi-generator-cli

# Generate TypeScript client
npx openapi-generator-cli generate \
  -i http://localhost:8000/openapi.json \
  -g typescript-axios \
  -o src/api/generated
```

### React Integration Example

```typescript
// src/hooks/useCausalGraphs.ts
import { useQuery, useMutation } from '@tanstack/react-query';
import { CausalGraphsApi, CausalGraphRequest } from '../api/generated';

const api = new CausalGraphsApi({
  basePath: process.env.REACT_APP_API_BASE_URL
});

export const useCausalGraphs = () => {
  return useQuery({
    queryKey: ['causal-graphs'],
    queryFn: () => api.getCausalGraphs()
  });
};

export const useCreateCausalGraph = () => {
  return useMutation({
    mutationFn: (graph: CausalGraphRequest) => 
      api.createCausalGraph(graph),
    onSuccess: () => {
      // Invalidate and refetch graphs
      queryClient.invalidateQueries({ queryKey: ['causal-graphs'] });
    }
  });
};
```

## Testing API Documentation

### Automated Tests

```python
# tests/test_api_docs.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_openapi_spec_generation():
    """Test that OpenAPI spec is generated correctly"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    spec = response.json()
    assert spec["info"]["title"] == "Causal UI Gym API"
    assert spec["info"]["version"] == "1.0.0"
    assert "paths" in spec
    assert "components" in spec

def test_swagger_ui_accessible():
    """Test that Swagger UI is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()

def test_redoc_accessible():
    """Test that ReDoc is accessible"""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "redoc" in response.text.lower()

@pytest.mark.parametrize("endpoint", [
    "/api/v1/graphs",
    "/api/v1/experiments",
    "/api/v1/analysis"
])
def test_endpoint_documentation(endpoint):
    """Test that all endpoints have proper documentation"""
    response = client.get("/openapi.json")
    spec = response.json()
    
    assert endpoint in spec["paths"]
    path_spec = spec["paths"][endpoint]
    
    # Check that POST method has description
    if "post" in path_spec:
        assert "description" in path_spec["post"]
        assert "responses" in path_spec["post"]
```

### Documentation Quality Checks

```python
# scripts/validate_docs.py
import requests
import json
from typing import Dict, List

def validate_openapi_spec(spec_url: str) -> Dict[str, List[str]]:
    """Validate OpenAPI specification for completeness"""
    response = requests.get(spec_url)
    spec = response.json()
    
    issues = {"errors": [], "warnings": []}
    
    # Check for required fields
    if not spec.get("info", {}).get("description"):
        issues["warnings"].append("Missing API description")
    
    if not spec.get("info", {}).get("contact"):
        issues["warnings"].append("Missing contact information")
    
    # Check paths
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if not details.get("description"):
                issues["warnings"].append(f"{method.upper()} {path}: Missing description")
            
            if not details.get("responses"):
                issues["errors"].append(f"{method.upper()} {path}: Missing response definitions")
    
    return issues

if __name__ == "__main__":
    issues = validate_openapi_spec("http://localhost:8000/openapi.json")
    
    if issues["errors"]:
        print("❌ Errors found:")
        for error in issues["errors"]:
            print(f"  - {error}")
    
    if issues["warnings"]:
        print("⚠️ Warnings:")
        for warning in issues["warnings"]:
            print(f"  - {warning}")
    
    if not issues["errors"] and not issues["warnings"]:
        print("✅ API documentation validation passed!")
```

## Maintenance

### Automation

Add to CI/CD pipeline:

```yaml
# .github/workflows/docs.yml (template)
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest requests
      
      - name: Start API server
        run: |
          uvicorn main:app --host 0.0.0.0 --port 8000 &
          sleep 10  # Wait for server to start
      
      - name: Validate API documentation
        run: |
          python scripts/validate_docs.py
          pytest tests/test_api_docs.py
      
      - name: Generate OpenAPI spec
        run: python scripts/generate_docs.py
      
      - name: Upload documentation artifacts
        uses: actions/upload-artifact@v3
        with:
          name: api-documentation
          path: docs/openapi.json
```

### Versioning

Track API versions in documentation:

```python
# Version management
API_VERSION = "1.0.0"
API_CHANGELOG = {
    "1.0.0": {
        "date": "2025-01-31",
        "changes": [
            "Initial API release",
            "Causal graph management endpoints",
            "Experiment tracking functionality"
        ]
    }
}

@app.get("/api/v1/version")
async def get_api_version():
    """Get current API version and changelog"""
    return {
        "version": API_VERSION,
        "changelog": API_CHANGELOG
    }
```

## Best Practices

1. **Comprehensive Documentation**: Every endpoint should have detailed descriptions
2. **Response Models**: Use Pydantic models for all responses
3. **Error Handling**: Standardized error responses with codes
4. **Examples**: Include realistic examples in all schemas
5. **Testing**: Automated tests for documentation completeness
6. **Versioning**: Clear API versioning strategy
7. **Client Generation**: Automated client library generation

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/tutorial/metadata/)
- [OpenAPI Specification](https://swagger.io/specification/)
- [Pydantic Models](https://pydantic-docs.helpmanual.io/)

---

*Last Updated: January 2025*  
*Review: Required for API endpoint implementation*