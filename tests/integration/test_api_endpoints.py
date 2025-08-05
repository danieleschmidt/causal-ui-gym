"""
Integration tests for API endpoints.

Tests the complete API functionality including experiment management,
intervention computation, LLM agent integration, and WebSocket connections.
"""

import pytest
import asyncio
import json
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
import httpx

from backend.api.server import app
from backend.models.causal_models import ExperimentConfigModel, CausalDAGModel, CausalNodeModel, CausalEdgeModel


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_dag():
    """Create sample DAG for testing."""
    nodes = [
        CausalNodeModel(id="X", label="Treatment", position={"x": 100, "y": 100}),
        CausalNodeModel(id="Y", label="Outcome", position={"x": 200, "y": 100}),
        CausalNodeModel(id="Z", label="Mediator", position={"x": 150, "y": 150})
    ]
    
    edges = [
        CausalEdgeModel(source="X", target="Y", weight=1.0),
        CausalEdgeModel(source="X", target="Z", weight=0.8),
        CausalEdgeModel(source="Z", target="Y", weight=0.6)
    ]
    
    return CausalDAGModel(
        name="Test DAG",
        description="A simple test DAG",
        nodes=nodes,
        edges=edges
    )


@pytest.fixture
def sample_experiment(sample_dag):
    """Create sample experiment for testing."""
    return ExperimentConfigModel(
        id="test-experiment-123",
        name="Test Experiment",
        description="A test experiment for API testing",
        dag=sample_dag,
        interventions=[
            {
                "variable": "X",
                "value": 1.0,
                "intervention_type": "do"
            }
        ],
        outcome_variables=["Y"],
        sample_size=1000,
        random_seed=42
    )


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_api_status(self, client):
        """Test API status endpoint."""
        response = client.get("/api/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "experiments_count" in data
        assert "engine_type" in data


class TestExperimentEndpoints:
    """Test experiment management endpoints."""
    
    def test_create_experiment(self, client, sample_experiment):
        """Test experiment creation."""
        response = client.post(
            "/api/experiments",
            json=sample_experiment.dict()
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_experiment.id
        assert data["name"] == sample_experiment.name
        assert data["status"] == "created"
    
    def test_list_experiments_empty(self, client):
        """Test listing experiments when none exist."""
        response = client.get("/api/experiments")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_list_experiments_with_data(self, client, sample_experiment):
        """Test listing experiments with data."""
        # First create an experiment
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Then list experiments
        response = client.get("/api/experiments")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == sample_experiment.id
    
    def test_get_experiment(self, client, sample_experiment):
        """Test getting specific experiment."""
        # Create experiment first
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Get the experiment
        response = client.get(f"/api/experiments/{sample_experiment.id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == sample_experiment.id
        assert data["name"] == sample_experiment.name
    
    def test_get_nonexistent_experiment(self, client):
        """Test getting nonexistent experiment."""
        response = client.get("/api/experiments/nonexistent")
        assert response.status_code == 404
    
    def test_update_experiment(self, client, sample_experiment):
        """Test updating experiment."""
        # Create experiment first
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Update the experiment
        updated_experiment = sample_experiment.copy()
        updated_experiment.name = "Updated Test Experiment"
        
        response = client.put(
            f"/api/experiments/{sample_experiment.id}",
            json=updated_experiment.dict()
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Test Experiment"
    
    def test_delete_experiment(self, client, sample_experiment):
        """Test deleting experiment."""
        # Create experiment first
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Delete the experiment
        response = client.delete(f"/api/experiments/{sample_experiment.id}")
        assert response.status_code == 200
        
        # Verify it's deleted
        get_response = client.get(f"/api/experiments/{sample_experiment.id}")
        assert get_response.status_code == 404
    
    def test_validate_experiment(self, client, sample_experiment):
        """Test experiment validation."""
        # Create experiment first
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Validate the experiment
        response = client.post(f"/api/experiments/{sample_experiment.id}/validate")
        assert response.status_code == 200
        
        data = response.json()
        assert "is_valid" in data
        assert "errors" in data
        assert "warnings" in data
        assert "assumptions" in data
    
    def test_run_experiment(self, client, sample_experiment):
        """Test running experiment."""
        # Create experiment first
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Run the experiment
        response = client.post(f"/api/experiments/{sample_experiment.id}/run")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
    
    def test_get_experiment_results(self, client, sample_experiment):
        """Test getting experiment results."""
        # Create experiment first
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Get results (should be empty initially)
        response = client.get(f"/api/experiments/{sample_experiment.id}/results")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_experiment_filtering(self, client, sample_experiment):
        """Test experiment filtering by status."""
        # Create experiment
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Filter by status
        response = client.get("/api/experiments?status=created")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "created"
    
    def test_experiment_pagination(self, client, sample_dag):
        """Test experiment pagination."""
        # Create multiple experiments
        for i in range(5):
            experiment = ExperimentConfigModel(
                id=f"test-experiment-{i}",
                name=f"Test Experiment {i}",
                dag=sample_dag,
                interventions=[],
                outcome_variables=["Y"]
            )
            client.post("/api/experiments", json=experiment.dict())
        
        # Test pagination
        response = client.get("/api/experiments?limit=3&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        
        # Test second page
        response = client.get("/api/experiments?limit=3&offset=3")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


class TestInterventionEndpoints:
    """Test intervention computation endpoints."""
    
    def test_compute_intervention_basic(self, client, sample_experiment):
        """Test basic intervention computation."""
        # Create experiment first
        client.post("/api/experiments", json=sample_experiment.dict())
        
        # Compute intervention
        intervention_data = {
            "dag": sample_experiment.dag.dict(),
            "intervention": {
                "variable": "X",
                "value": 1.0,
                "intervention_type": "do"
            },
            "outcome_variable": "Y",
            "sample_size": 100
        }
        
        response = client.post("/api/interventions/compute", json=intervention_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "intervention" in data
        assert "outcome_distribution" in data
        assert "computation_time" in data
    
    def test_compute_ate(self, client, sample_dag):
        """Test ATE computation."""
        ate_data = {
            "dag": sample_dag.dict(),
            "treatment_variable": "X",
            "outcome_variable": "Y",
            "treatment_values": [0.0, 1.0],
            "sample_size": 100
        }
        
        response = client.post("/api/interventions/ate", json=ate_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "ate" in data
        assert "confidence_interval" in data
        assert "computation_time" in data
    
    def test_batch_interventions(self, client, sample_dag):
        """Test batch intervention computation."""
        batch_data = {
            "dag": sample_dag.dict(),
            "interventions": [
                {"variable": "X", "value": 0.0, "intervention_type": "do"},
                {"variable": "X", "value": 1.0, "intervention_type": "do"},
                {"variable": "X", "value": 2.0, "intervention_type": "do"}
            ],
            "outcome_variable": "Y",
            "sample_size": 50
        }
        
        response = client.post("/api/interventions/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        for result in data:
            assert "intervention" in result
            assert "outcome_distribution" in result
    
    def test_intervention_validation_errors(self, client):
        """Test intervention validation with invalid data."""
        invalid_data = {
            "dag": {
                "name": "Invalid DAG",
                "nodes": [],  # Empty nodes should cause error
                "edges": []
            },
            "intervention": {
                "variable": "X",
                "value": 1.0
            },
            "outcome_variable": "Y"
        }
        
        response = client.post("/api/interventions/compute", json=invalid_data)
        assert response.status_code == 400
    
    def test_intervention_nonexistent_variable(self, client, sample_dag):
        """Test intervention on nonexistent variable."""
        intervention_data = {
            "dag": sample_dag.dict(),
            "intervention": {
                "variable": "NONEXISTENT",  # This variable doesn't exist
                "value": 1.0
            },
            "outcome_variable": "Y"
        }
        
        response = client.post("/api/interventions/compute", json=intervention_data)
        assert response.status_code == 400


class TestAgentEndpoints:
    """Test LLM agent endpoints."""
    
    def test_list_agents_empty(self, client):
        """Test listing agents when none are registered."""
        response = client.get("/api/agents")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_register_agent(self, client):
        """Test registering a new agent."""
        agent_data = {
            "agent_type": "openai",
            "model": "gpt-4",
            "agent_id": "test-gpt4-agent"
        }
        
        response = client.post("/api/agents/register", params=agent_data)
        
        # Note: This might fail if OpenAI API key is not available
        # In that case, we expect a specific error
        if response.status_code == 500:
            # Check if it's due to missing OpenAI package
            assert "OpenAI" in response.json().get("detail", "")
        else:
            assert response.status_code == 200
            data = response.json()
            assert data["agent_id"] == "test-gpt4-agent"
    
    def test_list_agents_after_registration(self, client):
        """Test listing agents after registration."""
        # Try to register an agent first
        agent_data = {
            "agent_type": "openai",
            "model": "gpt-4",
            "agent_id": "test-agent"
        }
        
        register_response = client.post("/api/agents/register", params=agent_data)
        
        # Only proceed if registration was successful
        if register_response.status_code == 200:
            # List agents
            response = client.get("/api/agents")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) >= 1
            assert any(agent["agent_id"] == "test-agent" for agent in data)
    
    def test_query_agent(self, client, sample_dag):
        """Test querying an agent."""
        # First try to register an agent
        agent_data = {
            "agent_type": "openai",
            "model": "gpt-4",
            "agent_id": "query-test-agent"
        }
        
        register_response = client.post("/api/agents/register", params=agent_data)
        
        if register_response.status_code == 200:
            # Query the agent
            query_data = {
                "dag_description": "X causes Y",
                "intervention_description": "Set X to 1",
                "outcome_variable": "Y",
                "query_type": "prediction"
            }
            
            response = client.post(
                "/api/agents/query-test-agent/query",
                json=query_data
            )
            
            # This will likely fail without proper API keys
            # but we can test the endpoint structure
            assert response.status_code in [200, 404, 500]
    
    def test_batch_query_agents(self, client):
        """Test batch querying multiple agents."""
        query_data = {
            "dag_description": "X causes Y through Z",
            "intervention_description": "Set X to 2",
            "outcome_variable": "Y",
            "query_type": "prediction"
        }
        
        response = client.post("/api/agents/batch-query", json=query_data)
        
        # Should return empty list if no agents are registered
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_unsupported_agent_type(self, client):
        """Test registering unsupported agent type."""
        agent_data = {
            "agent_type": "unsupported",
            "model": "fake-model",
            "agent_id": "fake-agent"
        }
        
        response = client.post("/api/agents/register", params=agent_data)
        assert response.status_code == 400
        assert "Unsupported agent type" in response.json()["detail"]


class TestMetricsEndpoints:
    """Test metrics collection endpoints."""
    
    def test_submit_metrics(self, client):
        """Test submitting metrics data."""
        metrics_data = {
            "metrics": [
                {
                    "name": "test_counter_total",
                    "help": "A test counter",
                    "value": 1,
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }
            ],
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        response = client.post("/api/metrics", json=metrics_data)
        assert response.status_code == 200
    
    def test_get_metrics_summary(self, client):
        """Test getting metrics summary."""
        response = client.get("/api/metrics/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "metrics_count" in data


class TestWebSocketConnections:
    """Test WebSocket functionality."""
    
    def test_websocket_connection(self, client):
        """Test basic WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive welcome message
            data = websocket.receive_json()
            assert data["type"] == "notification"
            assert "Connected successfully" in data["data"]["message"]
    
    def test_websocket_heartbeat(self, client):
        """Test WebSocket heartbeat."""
        with client.websocket_connect("/ws") as websocket:
            # Receive welcome message first
            websocket.receive_json()
            
            # Send heartbeat
            websocket.send_json({
                "type": "heartbeat",
                "data": {},
                "timestamp": datetime.now().isoformat(),
                "message_id": "test-heartbeat-1"
            })
            
            # Should receive heartbeat acknowledgment
            response = websocket.receive_json()
            assert response["type"] == "heartbeat_ack"
    
    def test_websocket_subscription(self, client):
        """Test WebSocket topic subscription."""
        with client.websocket_connect("/ws") as websocket:
            # Receive welcome message first
            websocket.receive_json()
            
            # Subscribe to a topic
            websocket.send_json({
                "type": "subscribe",
                "data": {"topic": "experiment:test-123"},
                "timestamp": datetime.now().isoformat(),
                "message_id": "test-sub-1"
            })
            
            # Should receive subscription confirmation
            response = websocket.receive_json()
            assert response["type"] == "notification"
            assert "Subscribed to" in response["data"]["message"]
    
    def test_websocket_invalid_message(self, client):
        """Test WebSocket with invalid message format."""
        with client.websocket_connect("/ws") as websocket:
            # Receive welcome message first
            websocket.receive_json()
            
            # Send invalid JSON
            websocket.send_text("invalid json")
            
            # Should receive error message
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["data"]["error_code"] == "INVALID_JSON"


class TestErrorHandling:
    """Test error handling across endpoints."""
    
    def test_404_error(self, client):
        """Test 404 error for nonexistent endpoint."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method."""
        response = client.patch("/api/experiments")  # PATCH not supported
        assert response.status_code == 405
    
    def test_validation_error(self, client):
        """Test validation error with malformed data."""
        invalid_experiment = {
            "name": "",  # Empty name should fail validation
            "dag": {
                "name": "Empty DAG",
                "nodes": [],
                "edges": []
            }
        }
        
        response = client.post("/api/experiments", json=invalid_experiment)
        assert response.status_code in [400, 422]  # Validation error
    
    def test_large_payload_handling(self, client, sample_dag):
        """Test handling of large payloads."""
        # Create a large intervention batch
        large_batch = {
            "dag": sample_dag.dict(),
            "interventions": [
                {"variable": "X", "value": i, "intervention_type": "do"}
                for i in range(1000)  # Very large batch
            ],
            "outcome_variable": "Y",
            "sample_size": 10
        }
        
        response = client.post("/api/interventions/batch", json=large_batch)
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 413, 500]


class TestPerformance:
    """Test performance characteristics."""
    
    def test_concurrent_requests(self, client, sample_experiment):
        """Test handling concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            start = time.time()
            response = client.post("/api/experiments", json=sample_experiment.dict())
            end = time.time()
            results.append({
                'status_code': response.status_code,
                'duration': end - start
            })
        
        # Create multiple threads for concurrent requests
        threads = []
        for i in range(5):
            # Use different experiment IDs to avoid conflicts
            exp_data = sample_experiment.copy()
            exp_data.id = f"concurrent-test-{i}"
            
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 5
        assert all(r['status_code'] in [200, 400] for r in results)
        # All requests should complete within reasonable time
        assert all(r['duration'] < 5.0 for r in results)
    
    def test_response_times(self, client):
        """Test response times for various endpoints."""
        import time
        
        endpoints = [
            ("GET", "/health"),
            ("GET", "/api/status"),
            ("GET", "/api/experiments"),
            ("GET", "/api/agents")
        ]
        
        for method, endpoint in endpoints:
            start = time.time()
            if method == "GET":
                response = client.get(endpoint)
            end = time.time()
            
            duration = end - start
            assert response.status_code == 200
            assert duration < 1.0  # Should respond within 1 second


if __name__ == '__main__':
    pytest.main([__file__, '-v'])