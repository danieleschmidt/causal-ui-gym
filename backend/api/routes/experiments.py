"""
Experiment management API endpoints.

This module provides REST API endpoints for creating, managing, and
retrieving causal reasoning experiments.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ...models.causal_models import (
    ExperimentConfigModel, 
    ExperimentStatus,
    CausalDAGModel,
    ValidationResult,
    CausalResultModel
)
from ..server import get_causal_engine, get_app_state
from ...engine.causal_engine import JaxCausalEngine, CausalDAG

logger = logging.getLogger(__name__)
router = APIRouter()


def convert_dag_model_to_engine(dag_model: CausalDAGModel) -> CausalDAG:
    """Convert Pydantic DAG model to engine DAG format."""
    nodes = [node.id for node in dag_model.nodes]
    edges = [(edge.source, edge.target) for edge in dag_model.edges]
    edge_weights = {(edge.source, edge.target): edge.weight for edge in dag_model.edges}
    
    # For now, use empty node data (in real implementation, this would come from data source)
    node_data = {node: [] for node in nodes}
    
    return CausalDAG(
        nodes=nodes,
        edges=edges,
        node_data=node_data,
        edge_weights=edge_weights
    )


@router.post("/experiments", response_model=ExperimentConfigModel)
async def create_experiment(
    experiment: ExperimentConfigModel,
    app_state: Dict[str, Any] = Depends(get_app_state),
    engine: JaxCausalEngine = Depends(get_causal_engine)
) -> ExperimentConfigModel:
    """
    Create a new causal reasoning experiment.
    
    Args:
        experiment: Experiment configuration
        
    Returns:
        Created experiment with assigned ID
    """
    try:
        # Validate the DAG structure
        dag = convert_dag_model_to_engine(experiment.dag)
        assumptions = engine.validate_dag_assumptions(dag)
        
        if not assumptions.get('is_acyclic', False):
            raise HTTPException(
                status_code=400, 
                detail="DAG contains cycles and is not valid"
            )
        
        # Store experiment in memory (in production, use proper database)
        experiment.status = ExperimentStatus.CREATED
        experiment.updated_at = datetime.utcnow()
        
        app_state["experiments"][experiment.id] = experiment
        
        logger.info(f"Created experiment: {experiment.id} - {experiment.name}")
        
        return experiment
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=List[ExperimentConfigModel])
async def list_experiments(
    status: Optional[ExperimentStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> List[ExperimentConfigModel]:
    """
    Get list of experiments with optional filtering.
    
    Args:
        status: Optional status filter
        limit: Maximum number of results
        offset: Pagination offset
        
    Returns:
        List of experiments matching criteria
    """
    experiments = list(app_state["experiments"].values())
    
    # Filter by status if provided
    if status:
        experiments = [exp for exp in experiments if exp.status == status]
    
    # Sort by creation date (newest first)
    experiments.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    paginated = experiments[offset:offset + limit]
    
    logger.info(f"Retrieved {len(paginated)} experiments (total: {len(experiments)})")
    
    return paginated


@router.get("/experiments/{experiment_id}", response_model=ExperimentConfigModel)
async def get_experiment(
    experiment_id: str,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> ExperimentConfigModel:
    """
    Get a specific experiment by ID.
    
    Args:
        experiment_id: Unique experiment identifier
        
    Returns:
        Experiment configuration
    """
    experiment = app_state["experiments"].get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return experiment


@router.put("/experiments/{experiment_id}", response_model=ExperimentConfigModel)
async def update_experiment(
    experiment_id: str,
    experiment_update: ExperimentConfigModel,
    app_state: Dict[str, Any] = Depends(get_app_state),
    engine: JaxCausalEngine = Depends(get_causal_engine)
) -> ExperimentConfigModel:
    """
    Update an existing experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        experiment_update: Updated experiment configuration
        
    Returns:
        Updated experiment configuration
    """
    existing_experiment = app_state["experiments"].get(experiment_id)
    
    if not existing_experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Don't allow updating completed experiments
    if existing_experiment.status == ExperimentStatus.COMPLETED:
        raise HTTPException(
            status_code=409, 
            detail="Cannot update completed experiment"
        )
    
    try:
        # Validate updated DAG
        dag = convert_dag_model_to_engine(experiment_update.dag)
        assumptions = engine.validate_dag_assumptions(dag)
        
        if not assumptions.get('is_acyclic', False):
            raise HTTPException(
                status_code=400,
                detail="Updated DAG contains cycles and is not valid"
            )
        
        # Preserve original ID and creation time
        experiment_update.id = experiment_id
        experiment_update.created_at = existing_experiment.created_at
        experiment_update.updated_at = datetime.utcnow()
        
        # Update stored experiment
        app_state["experiments"][experiment_id] = experiment_update
        
        logger.info(f"Updated experiment: {experiment_id}")
        
        return experiment_update
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> Dict[str, str]:
    """
    Delete an experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        
    Returns:
        Deletion confirmation
    """
    experiment = app_state["experiments"].get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Remove experiment from storage
    del app_state["experiments"][experiment_id]
    
    # Also remove any associated results
    results_to_remove = [
        result_id for result_id, result in app_state["results"].items()
        if result.dag_id == experiment_id
    ]
    
    for result_id in results_to_remove:
        del app_state["results"][result_id]
    
    logger.info(f"Deleted experiment: {experiment_id} and {len(results_to_remove)} results")
    
    return {"message": f"Experiment {experiment_id} deleted successfully"}


@router.post("/experiments/{experiment_id}/validate", response_model=ValidationResult)
async def validate_experiment(
    experiment_id: str,
    app_state: Dict[str, Any] = Depends(get_app_state),
    engine: JaxCausalEngine = Depends(get_causal_engine)
) -> ValidationResult:
    """
    Validate an experiment's causal assumptions and structure.
    
    Args:
        experiment_id: Unique experiment identifier
        
    Returns:
        Validation results with errors, warnings, and assumption checks
    """
    experiment = app_state["experiments"].get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    try:
        dag = convert_dag_model_to_engine(experiment.dag)
        assumptions = engine.validate_dag_assumptions(dag)
        
        errors = []
        warnings = []
        
        # Check for critical issues
        if not assumptions.get('is_acyclic', False):
            errors.append("DAG contains cycles")
        
        if not assumptions.get('is_connected', False):
            warnings.append("DAG is not connected - some variables may be isolated")
        
        if not assumptions.get('has_sufficient_data', False):
            warnings.append("Insufficient data for reliable causal inference")
        
        # Check intervention validity
        dag_nodes = {node.id for node in experiment.dag.nodes}
        for intervention in experiment.interventions:
            if intervention.variable not in dag_nodes:
                errors.append(f"Intervention variable '{intervention.variable}' not in DAG")
        
        # Check outcome variables
        for outcome in experiment.outcome_variables:
            if outcome not in dag_nodes:
                errors.append(f"Outcome variable '{outcome}' not in DAG")
        
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            assumptions=assumptions
        )
        
        logger.info(f"Validated experiment {experiment_id}: valid={is_valid}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating experiment {experiment_id}: {e}")
        return ValidationResult(
            is_valid=False,
            errors=[f"Validation error: {str(e)}"],
            warnings=[],
            assumptions={}
        )


@router.post("/experiments/{experiment_id}/run")
async def run_experiment(
    experiment_id: str,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> Dict[str, str]:
    """
    Start running an experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        
    Returns:
        Execution status
    """
    experiment = app_state["experiments"].get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED]:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment is already {experiment.status.value}"
        )
    
    # Update experiment status
    experiment.status = ExperimentStatus.RUNNING
    experiment.updated_at = datetime.utcnow()
    
    app_state["experiments"][experiment_id] = experiment
    
    logger.info(f"Started experiment: {experiment_id}")
    
    # In a real implementation, this would trigger background processing
    # For now, just return success
    return {"message": f"Experiment {experiment_id} started successfully"}


@router.get("/experiments/{experiment_id}/results", response_model=List[CausalResultModel])
async def get_experiment_results(
    experiment_id: str,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> List[CausalResultModel]:
    """
    Get all results for a specific experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        
    Returns:
        List of causal computation results
    """
    experiment = app_state["experiments"].get(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Find all results associated with this experiment
    results = [
        result for result in app_state["results"].values()
        if result.dag_id == experiment_id
    ]
    
    # Sort by creation time
    results.sort(key=lambda x: x.created_at, reverse=True)
    
    logger.info(f"Retrieved {len(results)} results for experiment {experiment_id}")
    
    return results