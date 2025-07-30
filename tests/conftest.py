"""Pytest configuration and shared fixtures."""

import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session")
def jax_config():
    """Configure JAX for testing."""
    # Disable JIT compilation for testing
    jax.config.update('jax_disable_jit', True)
    # Use CPU for consistent testing
    jax.config.update('jax_platform_name', 'cpu')
    yield
    # Reset configuration
    jax.config.update('jax_disable_jit', False)


@pytest.fixture
def sample_causal_dag():
    """Sample causal DAG for testing."""
    return {
        'nodes': ['X', 'Y', 'Z'],
        'edges': [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')],
        'node_data': {
            'X': {'type': 'continuous', 'range': [0, 100]},
            'Y': {'type': 'continuous', 'range': [0, 50]},
            'Z': {'type': 'continuous', 'range': [0, 25]}
        }
    }


@pytest.fixture
def sample_intervention():
    """Sample intervention for testing."""
    return {'X': 42.0}


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    with patch('openai.OpenAI') as mock_openai, \
         patch('anthropic.Anthropic') as mock_anthropic:
        
        # Mock OpenAI responses
        mock_openai.return_value.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Causal reasoning response"))
        ]
        
        # Mock Anthropic responses
        mock_anthropic.return_value.messages.create.return_value.content = [
            MagicMock(text="Causal reasoning response")
        ]
        
        yield {
            'openai': mock_openai.return_value,
            'anthropic': mock_anthropic.return_value
        }


@pytest.fixture
def sample_data():
    """Generate sample data for causal analysis."""
    np.random.seed(42)  # For reproducible tests
    n_samples = 1000
    
    # Generate correlated data simulating causal relationships
    x = np.random.normal(0, 1, n_samples)
    y = 2 * x + np.random.normal(0, 0.5, n_samples)  # Y caused by X
    z = y + 0.5 * x + np.random.normal(0, 0.3, n_samples)  # Z caused by X and Y
    
    return {
        'X': x,
        'Y': y,
        'Z': z
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    return str(test_dir)


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    
    # Mock API keys to prevent accidental real API calls
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def performance_benchmark():
    """Benchmark fixture for performance testing."""
    import time
    
    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.end_time - self.start_time
        
        def assert_under(self, max_seconds):
            duration = self.stop()
            assert duration < max_seconds, f"Operation took {duration:.3f}s, expected under {max_seconds}s"
    
    return PerformanceBenchmark()


@pytest.fixture
def gpu_available():
    """Check if GPU is available for tests."""
    try:
        import jax
        devices = jax.devices()
        return any(device.device_kind == 'gpu' for device in devices)
    except:
        return False


# Skip markers for conditional tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "llm: mark test as requiring LLM API"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    skip_slow = pytest.mark.skip(reason="slow tests skipped (use -m slow to run)")
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_llm = pytest.mark.skip(reason="LLM API not configured")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("-m") == "slow":
            item.add_marker(skip_slow)
        
        if "gpu" in item.keywords:
            try:
                import jax
                devices = jax.devices()
                if not any(device.device_kind == 'gpu' for device in devices):
                    item.add_marker(skip_gpu)
            except:
                item.add_marker(skip_gpu)
        
        if "llm" in item.keywords:
            if not (os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY")):
                if not os.getenv("TESTING"):  # Allow mocked tests in testing mode
                    item.add_marker(skip_llm)