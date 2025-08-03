"""
Intervention processing API endpoints.

This module provides REST API endpoints for performing causal interventions
and computing their effects using the JAX causal engine.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any
import logging
import asyncio
from datetime import datetime

from ...models.causal_models import (
    InterventionModel,
    CausalResultModel,
    CausalMetricModel,
    MetricType,
    ExperimentConfigModel
)
from ..server import get_causal_engine, get_app_state
from ...engine.causal_engine import JaxCausalEngine, CausalDAG, Intervention
from .experiments import convert_dag_model_to_engine

logger = logging.getLogger(__name__)
router = APIRouter()


def convert_intervention_model_to_engine(intervention_model: InterventionModel) -> Intervention:
    """Convert Pydantic intervention model to engine intervention format."""
    return Intervention(
        variable=intervention_model.variable,
        value=float(intervention_model.value),
        timestamp=intervention_model.timestamp.timestamp()
    )


async def compute_intervention_async(
    engine: JaxCausalEngine,
    dag: CausalDAG,
    intervention: Intervention,
    outcome_variable: str,
    sample_size: int = 10000
) -> CausalResultModel:
    """
    Asynchronously compute intervention effects.
    
    This function runs the computationally intensive causal inference
    in a background thread to avoid blocking the API.
    """
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        engine.compute_intervention,
        dag,
        intervention,
        outcome_variable,
        sample_size
    )
    
    # Convert engine result to API model
    metrics = []
    
    # Add ATE metric if available
    if result.ate is not None:
        ate_metric = CausalMetricModel(
            metric_type=MetricType.ATE,
            value=result.ate,
            confidence_interval=result.confidence_interval,
            sample_size=sample_size,
            computation_time=result.computation_time or 0.0
        )
        metrics.append(ate_metric)
    
    # Convert intervention back to model
    intervention_model = InterventionModel(
        variable=intervention.variable,
        value=intervention.value,
        intervention_type="do",
        timestamp=datetime.fromtimestamp(intervention.timestamp or datetime.utcnow().timestamp())
    )
    
    return CausalResultModel(
        dag_id="",  # Will be set by caller
        intervention=intervention_model,
        outcome_variable=outcome_variable,
        metrics=metrics,
        outcome_distribution=result.outcome_distribution.tolist() if len(result.outcome_distribution) > 0 else None
    )


@router.post("/interventions/single", response_model=CausalResultModel)
async def perform_single_intervention(
    experiment_id: str,
    intervention: InterventionModel,
    outcome_variable: str,
    sample_size: int = 10000,
    app_state: Dict[str, Any] = Depends(get_app_state),
    engine: JaxCausalEngine = Depends(get_causal_engine)
) -> CausalResultModel:
    """
    Perform a single causal intervention and compute its effects.
    
    Args:
        experiment_id: ID of experiment containing the DAG
        intervention: Intervention specification
        outcome_variable: Variable to analyze as outcome
        sample_size: Number of samples for Monte Carlo estimation
        
    Returns:
        Causal computation results
    """
    # Get experiment
    experiment = app_state["experiments"].get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Validate intervention variable exists in DAG
    dag_nodes = {node.id for node in experiment.dag.nodes}
    if intervention.variable not in dag_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Intervention variable '{intervention.variable}' not found in DAG"
        )
    
    # Validate outcome variable exists in DAG
    if outcome_variable not in dag_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Outcome variable '{outcome_variable}' not found in DAG"
        )
    
    try:
        # Convert models to engine format
        dag = convert_dag_model_to_engine(experiment.dag)
        engine_intervention = convert_intervention_model_to_engine(intervention)
        
        # Compute intervention effects
        result = await compute_intervention_async(
            engine, dag, engine_intervention, outcome_variable, sample_size
        )
        
        # Set DAG ID and store result
        result.dag_id = experiment_id
        app_state["results"][result.id] = result
        
        logger.info(
            f"Computed intervention on {intervention.variable}={intervention.value} "
            f"for outcome {outcome_variable} in experiment {experiment_id}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error computing intervention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interventions/batch", response_model=List[CausalResultModel])
async def perform_batch_interventions(
    experiment_id: str,
    interventions: List[InterventionModel],
    outcome_variable: str,
    sample_size: int = 10000,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    app_state: Dict[str, Any] = Depends(get_app_state),
    engine: JaxCausalEngine = Depends(get_causal_engine)
) -> List[CausalResultModel]:
    """
    Perform multiple causal interventions in parallel.
    
    Args:
        experiment_id: ID of experiment containing the DAG
        interventions: List of interventions to perform
        outcome_variable: Variable to analyze as outcome
        sample_size: Number of samples for Monte Carlo estimation
        
    Returns:
        List of causal computation results
    """
    # Get experiment
    experiment = app_state["experiments"].get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Validate all variables exist in DAG
    dag_nodes = {node.id for node in experiment.dag.nodes}
    
    for intervention in interventions:
        if intervention.variable not in dag_nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Intervention variable '{intervention.variable}' not found in DAG"
            )
    
    if outcome_variable not in dag_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Outcome variable '{outcome_variable}' not found in DAG"
        )
    
    try:
        # Convert models to engine format
        dag = convert_dag_model_to_engine(experiment.dag)
        engine_interventions = [
            convert_intervention_model_to_engine(intervention)
            for intervention in interventions
        ]
        
        # Compute all interventions in parallel
        tasks = [
            compute_intervention_async(engine, dag, intervention, outcome_variable, sample_size)
            for intervention in engine_interventions
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Set DAG IDs and store results
        for result in results:
            result.dag_id = experiment_id
            app_state["results"][result.id] = result
        
        logger.info(
            f"Computed {len(results)} batch interventions "
            f"for outcome {outcome_variable} in experiment {experiment_id}"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error computing batch interventions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interventions/ate", response_model=CausalMetricModel)
async def compute_average_treatment_effect(
    experiment_id: str,
    treatment_variable: str,
    outcome_variable: str,
    treatment_values: List[float] = [0.0, 1.0],
    sample_size: int = 10000,
    confidence_level: float = 0.95,
    app_state: Dict[str, Any] = Depends(get_app_state),
    engine: JaxCausalEngine = Depends(get_causal_engine)
) -> CausalMetricModel:
    """
    Compute Average Treatment Effect between treatment conditions.
    
    Args:
        experiment_id: ID of experiment containing the DAG
        treatment_variable: Name of treatment variable
        outcome_variable: Name of outcome variable
        treatment_values: Values of treatment to compare (default: [0, 1])
        sample_size: Number of samples for estimation
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        ATE metric with confidence intervals
    """
    # Get experiment
    experiment = app_state["experiments"].get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Validate variables exist in DAG
    dag_nodes = {node.id for node in experiment.dag.nodes}
    
    if treatment_variable not in dag_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Treatment variable '{treatment_variable}' not found in DAG"
        )
    
    if outcome_variable not in dag_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Outcome variable '{outcome_variable}' not found in DAG"
        )
    
    try:
        # Convert DAG to engine format
        dag = convert_dag_model_to_engine(experiment.dag)
        
        # Compute ATE using engine
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            engine.compute_ate,
            dag,
            treatment_variable,
            outcome_variable,
            treatment_values,
            sample_size,
            confidence_level
        )
        
        # Convert to metric model
        metric = CausalMetricModel(
            metric_type=MetricType.ATE,
            value=result.ate,
            confidence_interval=result.confidence_interval,
            sample_size=sample_size,
            computation_time=result.computation_time,
            metadata={
                "treatment_variable": treatment_variable,
                "outcome_variable": outcome_variable,
                "treatment_values": treatment_values,
                "confidence_level": confidence_level
            }
        )
        
        logger.info(
            f"Computed ATE for {treatment_variable} -> {outcome_variable}: {result.ate:.4f} "
            f"(CI: {result.confidence_interval}) in experiment {experiment_id}"
        )
        
        return metric
        
    except Exception as e:
        logger.error(f"Error computing ATE: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interventions/backdoor", response_model=Dict[str, Any])
async def identify_backdoor_paths(
    experiment_id: str,
    treatment_variable: str,
    outcome_variable: str,
    app_state: Dict[str, Any] = Depends(get_app_state),
    engine: JaxCausalEngine = Depends(get_causal_engine)
) -> Dict[str, Any]:
    """
    Identify backdoor paths between treatment and outcome variables.
    
    Args:
        experiment_id: ID of experiment containing the DAG
        treatment_variable: Name of treatment variable
        outcome_variable: Name of outcome variable
        
    Returns:
        Dictionary containing backdoor paths and blocking sets
    """
    # Get experiment
    experiment = app_state["experiments"].get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Validate variables exist in DAG
    dag_nodes = {node.id for node in experiment.dag.nodes}
    
    if treatment_variable not in dag_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Treatment variable '{treatment_variable}' not found in DAG"
        )
    
    if outcome_variable not in dag_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Outcome variable '{outcome_variable}' not found in DAG"
        )
    
    try:
        # Convert DAG to engine format
        dag = convert_dag_model_to_engine(experiment.dag)
        
        # Identify backdoor paths
        backdoor_paths = engine.identify_backdoor_paths(dag, treatment_variable, outcome_variable)
        
        # For each path, suggest potential blocking sets
        blocking_suggestions = []
        for path in backdoor_paths:
            if len(path) > 2:  # Path has intermediate nodes
                # Suggest intermediate nodes as potential confounders to control for
                intermediate_nodes = path[1:-1]  # Exclude treatment and outcome
                blocking_suggestions.append({
                    "path": path,
                    "suggested_controls": intermediate_nodes
                })
        
        result = {
            "treatment": treatment_variable,
            "outcome": outcome_variable,
            "backdoor_paths": backdoor_paths,
            "num_backdoor_paths": len(backdoor_paths),
            "blocking_suggestions": blocking_suggestions,
            "requires_adjustment": len(backdoor_paths) > 0
        }
        
        logger.info(
            f"Identified {len(backdoor_paths)} backdoor paths from {treatment_variable} "
            f"to {outcome_variable} in experiment {experiment_id}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error identifying backdoor paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interventions/{result_id}", response_model=CausalResultModel)
async def get_intervention_result(
    result_id: str,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> CausalResultModel:
    """
    Get a specific intervention result by ID.
    
    Args:
        result_id: Unique result identifier
        
    Returns:
        Causal computation result
    """
    result = app_state["results"].get(result_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Intervention result not found")
    
    return result


@router.delete("/interventions/{result_id}")
async def delete_intervention_result(
    result_id: str,
    app_state: Dict[str, Any] = Depends(get_app_state)
) -> Dict[str, str]:
    """
    Delete a specific intervention result.
    
    Args:
        result_id: Unique result identifier
        
    Returns:
        Deletion confirmation
    """
    result = app_state["results"].get(result_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Intervention result not found")
    
    del app_state["results"][result_id]
    
    logger.info(f"Deleted intervention result: {result_id}")
    
    return {"message": f"Intervention result {result_id} deleted successfully"}