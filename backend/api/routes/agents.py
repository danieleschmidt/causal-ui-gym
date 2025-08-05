"""
LLM Agent management API endpoints.

This module provides REST API endpoints for managing and interacting
with different LLM agents for causal reasoning evaluation.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pydantic import BaseModel

from ..server import get_app_state, get_causal_engine
from ...llm.llm_agents import (
    LLMAgentManager, 
    OpenAIAgent, 
    AnthropicAgent, 
    CausalQuery, 
    LLMResponse
)
from ...engine.causal_engine import CausalDAG, Intervention

logger = logging.getLogger(__name__)
router = APIRouter()

# Global agent manager
agent_manager = LLMAgentManager()


class AgentQueryRequest(BaseModel):
    dag_description: str
    intervention_description: str
    outcome_variable: str
    query_type: str = "prediction"
    context: Dict[str, Any] = {}


class AgentComparisonRequest(BaseModel):
    dag_id: str
    intervention_variable: str
    intervention_values: List[float]
    outcome_variable: str
    agent_ids: Optional[List[str]] = None


@router.get("/agents", response_model=List[Dict[str, Any]])
async def list_agents(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    status: Optional[str] = Query(None, description="Filter by status"),
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> List[Dict[str, Any]]:
    """
    Get list of available LLM agents.
    
    Args:
        provider: Optional provider filter (openai, anthropic, etc.)
        status: Optional status filter (available, busy, error)
        
    Returns:
        List of LLM agents matching criteria
    """
    # Get agents from manager
    agents = agent_manager.list_agents()
    
    # Apply filters
    if provider:
        agents = [agent for agent in agents if provider.lower() in agent["type"].lower()]
    
    if status:
        agents = [agent for agent in agents if agent["status"] == status]
    
    logger.info(f"Retrieved {len(agents)} agents")
    
    return agents


@router.post("/agents/register")
async def register_agent(
    agent_type: str,
    model: str,
    agent_id: Optional[str] = None,
    api_key: Optional[str] = None,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> Dict[str, Any]:
    """
    Register a new LLM agent.
    
    Args:
        agent_type: Type of agent (openai, anthropic)
        model: Model name (e.g., gpt-4, claude-3-haiku-20240307)
        agent_id: Optional custom agent ID
        api_key: Optional API key (if not set in environment)
        
    Returns:
        Registration confirmation with agent details
    """
    try:
        if agent_id is None:
            agent_id = f"{agent_type}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create agent based on type
        if agent_type.lower() == "openai":
            agent = OpenAIAgent(agent_id=agent_id, model=model, api_key=api_key)
        elif agent_type.lower() == "anthropic":
            agent = AnthropicAgent(agent_id=agent_id, model=model, api_key=api_key)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported agent type: {agent_type}")
        
        # Register with manager
        agent_manager.register_agent(agent)
        
        logger.info(f"Registered agent: {agent_id} - {agent_type} ({model})")
        
        return {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "model": model,
            "status": "registered",
            "message": f"Agent {agent_id} registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/query")
async def query_agent(
    agent_id: str,
    query_request: AgentQueryRequest,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> Dict[str, Any]:
    """
    Send a causal reasoning query to an agent.
    
    Args:
        agent_id: Unique agent identifier
        query_request: Causal reasoning query details
        
    Returns:
        Agent response with reasoning
    """
    agent = agent_manager.get_agent(agent_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if agent.status != "available":
        raise HTTPException(status_code=409, detail=f"Agent is {agent.status}")
    
    try:
        # Create causal query
        query = CausalQuery(
            dag_description=query_request.dag_description,
            intervention_description=query_request.intervention_description,
            outcome_variable=query_request.outcome_variable,
            query_type=query_request.query_type,
            context=query_request.context
        )
        
        # Query the agent
        response = await agent.query_causal_effect(query)
        
        logger.info(f"Processed query for agent: {agent_id}")
        
        return {
            "agent_id": response.agent_id,
            "model": response.model,
            "query": response.query,
            "response": response.response,
            "reasoning_steps": response.reasoning_steps,
            "predicted_effect": response.predicted_effect,
            "confidence": response.confidence,
            "timestamp": response.timestamp.isoformat(),
            "processing_time": response.processing_time,
            "metadata": response.metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing query for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/batch-query")
async def batch_query_agents(
    query_request: AgentQueryRequest,
    agent_ids: Optional[List[str]] = None,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> List[Dict[str, Any]]:
    """
    Query multiple agents with the same causal reasoning question.
    
    Args:
        query_request: Causal reasoning query details
        agent_ids: Optional list of specific agent IDs to query
        
    Returns:
        List of responses from all queried agents
    """
    try:
        # Create causal query
        query = CausalQuery(
            dag_description=query_request.dag_description,
            intervention_description=query_request.intervention_description,
            outcome_variable=query_request.outcome_variable,
            query_type=query_request.query_type,
            context=query_request.context
        )
        
        # Batch query agents
        responses = await agent_manager.batch_query(query, agent_ids)
        
        logger.info(f"Batch query completed with {len(responses)} responses")
        
        # Convert responses to dict format
        formatted_responses = []
        for response in responses:
            formatted_responses.append({
                "agent_id": response.agent_id,
                "model": response.model,
                "response": response.response,
                "reasoning_steps": response.reasoning_steps,
                "predicted_effect": response.predicted_effect,
                "confidence": response.confidence,
                "timestamp": response.timestamp.isoformat(),
                "processing_time": response.processing_time,
                "metadata": response.metadata
            })
        
        return formatted_responses
        
    except Exception as e:
        logger.error(f"Error in batch query: {e}")
        raise HTTPException(status_code=500, detail=str(e))