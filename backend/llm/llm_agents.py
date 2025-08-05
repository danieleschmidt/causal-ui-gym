"""
LLM Agent integration for causal reasoning evaluation.

This module provides interfaces for testing different LLM models on 
causal reasoning tasks and collecting their responses for analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..engine.causal_engine import CausalDAG, Intervention, CausalResult

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Container for LLM response to causal reasoning query."""
    agent_id: str
    model: str
    query: str
    response: str
    reasoning_steps: List[str]
    predicted_effect: Optional[float]
    confidence: float
    timestamp: datetime
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class CausalQuery:
    """Causal reasoning query for LLM evaluation."""
    dag_description: str
    intervention_description: str
    outcome_variable: str
    query_type: str  # 'prediction', 'explanation', 'identification'
    context: Dict[str, Any]


class BaseLLMAgent(ABC):
    """Base class for LLM agents that perform causal reasoning tasks."""
    
    def __init__(self, agent_id: str, model: str):
        self.agent_id = agent_id
        self.model = model
        self.status = "available"
        
    @abstractmethod
    async def query_causal_effect(self, query: CausalQuery) -> LLMResponse:
        """Query the LLM about a causal effect."""
        pass
        
    @abstractmethod
    async def explain_causal_mechanism(self, dag: CausalDAG, intervention: Intervention) -> LLMResponse:
        """Ask LLM to explain causal mechanisms."""
        pass
        
    @abstractmethod
    async def identify_confounders(self, dag: CausalDAG, treatment: str, outcome: str) -> LLMResponse:
        """Ask LLM to identify potential confounders."""
        pass


class OpenAIAgent(BaseLLMAgent):
    """OpenAI GPT agent for causal reasoning evaluation."""
    
    def __init__(self, agent_id: str, model: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(agent_id, model)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available")
        
        self.client = openai.AsyncClient(api_key=api_key)
        
    async def query_causal_effect(self, query: CausalQuery) -> LLMResponse:
        """Query GPT about causal effects."""
        start_time = datetime.now()
        
        prompt = self._build_causal_prompt(query)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            predicted_effect = self._extract_predicted_effect(content)
            reasoning_steps = self._extract_reasoning_steps(content)
            confidence = self._extract_confidence(content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=content,
                reasoning_steps=reasoning_steps,
                predicted_effect=predicted_effect,
                confidence=confidence,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "finish_reason": response.choices[0].finish_reason
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=f"Error: {str(e)}",
                reasoning_steps=[],
                predicted_effect=None,
                confidence=0.0,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    async def explain_causal_mechanism(self, dag: CausalDAG, intervention: Intervention) -> LLMResponse:
        """Ask GPT to explain causal mechanisms."""
        start_time = datetime.now()
        
        dag_description = self._describe_dag(dag)
        intervention_description = f"Setting {intervention.variable} to {intervention.value}"
        
        prompt = f"""
        Given the following causal graph:
        {dag_description}
        
        Explain the causal mechanism by which the intervention "{intervention_description}" 
        would affect other variables in the system. 
        
        Please provide:
        1. Step-by-step explanation of the causal pathway
        2. Which variables would be directly affected
        3. Which variables would be indirectly affected
        4. Any potential confounding factors to consider
        
        Structure your response clearly with numbered steps.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1200
            )
            
            content = response.choices[0].message.content
            reasoning_steps = self._extract_reasoning_steps(content)
            confidence = self._extract_confidence(content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=content,
                reasoning_steps=reasoning_steps,
                predicted_effect=None,
                confidence=confidence,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "query_type": "mechanism_explanation"
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error in mechanism explanation: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=f"Error: {str(e)}",
                reasoning_steps=[],
                predicted_effect=None,
                confidence=0.0,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    async def identify_confounders(self, dag: CausalDAG, treatment: str, outcome: str) -> LLMResponse:
        """Ask GPT to identify potential confounders."""
        start_time = datetime.now()
        
        dag_description = self._describe_dag(dag)
        
        prompt = f"""
        Given the following causal graph:
        {dag_description}
        
        Identify all potential confounders between the treatment variable "{treatment}" 
        and the outcome variable "{outcome}".
        
        A confounder is a variable that:
        1. Influences both the treatment and the outcome
        2. Creates a spurious association between treatment and outcome
        3. Must be controlled for to get unbiased causal estimates
        
        Please provide:
        1. List of confounding variables (if any)
        2. Explanation of why each is a confounder
        3. Suggested control strategy
        4. Assessment of whether the causal effect is identifiable
        
        Format your response with clear sections and confidence ratings.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            reasoning_steps = self._extract_reasoning_steps(content)
            confidence = self._extract_confidence(content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=content,
                reasoning_steps=reasoning_steps,
                predicted_effect=None,
                confidence=confidence,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "query_type": "confounder_identification",
                    "treatment": treatment,
                    "outcome": outcome
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error in confounder identification: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=f"Error: {str(e)}",
                reasoning_steps=[],
                predicted_effect=None,
                confidence=0.0,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    def _get_system_prompt(self) -> str:
        """Get system prompt for causal reasoning tasks."""
        return """
        You are an expert in causal inference and statistics. Your task is to analyze 
        causal relationships in directed acyclic graphs (DAGs) and provide accurate, 
        well-reasoned responses about causal effects.
        
        Key principles to follow:
        1. Carefully analyze the causal structure before making conclusions
        2. Distinguish between correlation and causation
        3. Consider confounding variables and selection bias
        4. Provide step-by-step reasoning for your conclusions
        5. Express uncertainty when appropriate
        6. Use proper causal inference terminology
        
        When providing numerical predictions, use the format: "Predicted effect: X.XX"
        When expressing confidence, use the format: "Confidence: XX%"
        Always structure your reasoning in clear, numbered steps.
        """

    def _build_causal_prompt(self, query: CausalQuery) -> str:
        """Build prompt for causal effect query."""
        return f"""
        Causal Graph Description:
        {query.dag_description}
        
        Intervention:
        {query.intervention_description}
        
        Question: What is the expected causal effect of this intervention on {query.outcome_variable}?
        
        Please provide:
        1. Your predicted numerical effect (if quantifiable)
        2. Step-by-step causal reasoning
        3. Confidence level in your prediction
        4. Any important assumptions or limitations
        
        Query Type: {query.query_type}
        Additional Context: {json.dumps(query.context, indent=2)}
        """

    def _describe_dag(self, dag: CausalDAG) -> str:
        """Convert DAG to text description for LLM."""
        description = f"Variables: {', '.join(dag.nodes)}\n\n"
        description += "Causal Relationships:\n"
        
        for edge in dag.edges:
            source, target = edge
            weight = dag.edge_weights.get(edge, 1.0)
            description += f"- {source} → {target} (strength: {weight:.2f})\n"
        
        return description

    def _extract_predicted_effect(self, response: str) -> Optional[float]:
        """Extract numerical prediction from LLM response."""
        import re
        
        # Look for patterns like "Predicted effect: 0.25" or "Effect size: -1.2"
        patterns = [
            r"predicted effect:\s*(-?\d+\.?\d*)",
            r"effect size:\s*(-?\d+\.?\d*)",
            r"estimate:\s*(-?\d+\.?\d*)",
            r"impact:\s*(-?\d+\.?\d*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence level from LLM response."""
        import re
        
        patterns = [
            r"confidence:\s*(\d+)%",
            r"confident:\s*(\d+)%",
            r"certainty:\s*(\d+)%"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    return float(match.group(1)) / 100.0
                except ValueError:
                    continue
        
        # Default moderate confidence if not specified
        return 0.7

    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract numbered reasoning steps from LLM response."""
        import re
        
        # Look for numbered lists
        steps = re.findall(r'^\d+\.\s*(.+)$', response, re.MULTILINE)
        
        if not steps:
            # Fallback: split by sentences
            sentences = response.split('.')
            steps = [s.strip() for s in sentences if len(s.strip()) > 10][:5]
        
        return steps


class AnthropicAgent(BaseLLMAgent):
    """Anthropic Claude agent for causal reasoning evaluation."""
    
    def __init__(self, agent_id: str, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        super().__init__(agent_id, model)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available")
            
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def query_causal_effect(self, query: CausalQuery) -> LLMResponse:
        """Query Claude about causal effects."""
        start_time = datetime.now()
        
        prompt = self._build_causal_prompt(query)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            predicted_effect = self._extract_predicted_effect(content)
            reasoning_steps = self._extract_reasoning_steps(content)
            confidence = self._extract_confidence(content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=content,
                reasoning_steps=reasoning_steps,
                predicted_effect=predicted_effect,
                confidence=confidence,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                    "stop_reason": response.stop_reason
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=f"Error: {str(e)}",
                reasoning_steps=[],
                predicted_effect=None,
                confidence=0.0,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    async def explain_causal_mechanism(self, dag: CausalDAG, intervention: Intervention) -> LLMResponse:
        """Ask Claude to explain causal mechanisms."""
        # Similar implementation to OpenAI but with Claude-specific formatting
        start_time = datetime.now()
        
        dag_description = self._describe_dag(dag)
        intervention_description = f"Setting {intervention.variable} to {intervention.value}"
        
        prompt = f"""
        You are an expert in causal inference. Given this causal graph:
        
        {dag_description}
        
        Explain how the intervention "{intervention_description}" would causally affect the system.
        
        Provide your analysis in this format:
        1. Direct effects (immediate causal impacts)
        2. Indirect effects (downstream consequences)  
        3. Potential confounders or mediators
        4. Overall assessment and confidence level
        
        Be precise and use causal reasoning principles.
        """
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1200,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            reasoning_steps = self._extract_reasoning_steps(content)
            confidence = self._extract_confidence(content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=content,
                reasoning_steps=reasoning_steps,
                predicted_effect=None,
                confidence=confidence,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                    "query_type": "mechanism_explanation"
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error in mechanism explanation: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=f"Error: {str(e)}",
                reasoning_steps=[],
                predicted_effect=None,
                confidence=0.0,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    async def identify_confounders(self, dag: CausalDAG, treatment: str, outcome: str) -> LLMResponse:
        """Ask Claude to identify potential confounders."""
        # Similar implementation to OpenAI with Claude-specific approach
        start_time = datetime.now()
        
        dag_description = self._describe_dag(dag)
        
        prompt = f"""
        As a causal inference expert, analyze this causal graph for confounders:
        
        {dag_description}
        
        Treatment variable: {treatment}
        Outcome variable: {outcome}
        
        Task: Identify variables that confound the relationship between treatment and outcome.
        
        For each potential confounder, explain:
        - Why it qualifies as a confounder
        - How it affects causal identification
        - What control strategy you recommend
        
        Conclude with your assessment of effect identifiability and confidence level.
        """
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            reasoning_steps = self._extract_reasoning_steps(content)
            confidence = self._extract_confidence(content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=content,
                reasoning_steps=reasoning_steps,
                predicted_effect=None,
                confidence=confidence,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                    "query_type": "confounder_identification",
                    "treatment": treatment,
                    "outcome": outcome
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error in confounder identification: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                agent_id=self.agent_id,
                model=self.model,
                query=prompt,
                response=f"Error: {str(e)}",
                reasoning_steps=[],
                predicted_effect=None,
                confidence=0.0,
                timestamp=start_time,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    def _build_causal_prompt(self, query: CausalQuery) -> str:
        """Build prompt for causal effect query (Claude version)."""
        return f"""
        You are Claude, an AI assistant with expertise in causal inference and statistics.
        
        Causal Graph:
        {query.dag_description}
        
        Intervention Scenario:
        {query.intervention_description}
        
        Question: Predict the causal effect of this intervention on {query.outcome_variable}.
        
        Provide your analysis following this structure:
        1. Causal pathway analysis
        2. Numerical effect prediction (if possible)
        3. Confidence assessment (as percentage)
        4. Key assumptions and limitations
        
        Query details: {query.query_type}
        Context: {json.dumps(query.context, indent=2)}
        
        Remember to distinguish causation from correlation and consider confounding.
        """

    def _describe_dag(self, dag: CausalDAG) -> str:
        """Convert DAG to text description for Claude."""
        description = f"Variables: {', '.join(dag.nodes)}\n\n"
        description += "Causal Structure:\n"
        
        for edge in dag.edges:
            source, target = edge
            weight = dag.edge_weights.get(edge, 1.0)
            description += f"• {source} causes {target} (effect strength: {weight:.2f})\n"
        
        return description

    def _extract_predicted_effect(self, response: str) -> Optional[float]:
        """Extract numerical prediction from Claude response."""
        # Reuse OpenAI extraction logic
        return OpenAIAgent._extract_predicted_effect(self, response)

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence level from Claude response."""
        # Reuse OpenAI extraction logic  
        return OpenAIAgent._extract_confidence(self, response)

    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from Claude response."""
        # Reuse OpenAI extraction logic
        return OpenAIAgent._extract_reasoning_steps(self, response)


class LLMAgentManager:
    """Manager for multiple LLM agents performing causal reasoning tasks."""
    
    def __init__(self):
        self.agents: Dict[str, BaseLLMAgent] = {}
        
    def register_agent(self, agent: BaseLLMAgent) -> None:
        """Register a new LLM agent."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered LLM agent: {agent.agent_id} ({agent.model})")
        
    def get_agent(self, agent_id: str) -> Optional[BaseLLMAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
        
    def list_agents(self) -> List[Dict[str, str]]:
        """List all registered agents."""
        return [
            {
                "agent_id": agent.agent_id,
                "model": agent.model,
                "status": agent.status,
                "type": type(agent).__name__
            }
            for agent in self.agents.values()
        ]
        
    async def batch_query(
        self,
        query: CausalQuery,
        agent_ids: Optional[List[str]] = None
    ) -> List[LLMResponse]:
        """Query multiple agents concurrently."""
        if agent_ids is None:
            target_agents = list(self.agents.values())
        else:
            target_agents = [self.agents[aid] for aid in agent_ids if aid in self.agents]
            
        if not target_agents:
            logger.warning("No valid agents found for batch query")
            return []
            
        logger.info(f"Batch querying {len(target_agents)} agents")
        
        tasks = [agent.query_causal_effect(query) for agent in target_agents]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Agent {target_agents[i].agent_id} failed: {response}")
            else:
                valid_responses.append(response)
                
        return valid_responses
        
    async def compare_agents_on_dag(
        self,
        dag: CausalDAG,
        interventions: List[Intervention],
        outcome_variable: str,
        agent_ids: Optional[List[str]] = None
    ) -> Dict[str, List[LLMResponse]]:
        """Compare multiple agents on the same causal reasoning task."""
        results = {}
        
        for intervention in interventions:
            query = CausalQuery(
                dag_description=self._describe_dag(dag),
                intervention_description=f"Setting {intervention.variable} to {intervention.value}",
                outcome_variable=outcome_variable,
                query_type="prediction",
                context={
                    "intervention_type": intervention.intervention_type,
                    "timestamp": str(intervention.timestamp)
                }
            )
            
            responses = await self.batch_query(query, agent_ids)
            results[f"{intervention.variable}={intervention.value}"] = responses
            
        return results
        
    def _describe_dag(self, dag: CausalDAG) -> str:
        """Convert DAG to text description."""
        description = f"Variables: {', '.join(dag.nodes)}\n\n"
        description += "Causal Relationships:\n"
        
        for edge in dag.edges:
            source, target = edge
            weight = dag.edge_weights.get(edge, 1.0)
            description += f"- {source} → {target} (strength: {weight:.2f})\n"
        
        return description