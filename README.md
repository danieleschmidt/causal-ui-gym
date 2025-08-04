# Causal UI Gym

[![React](https://img.shields.io/badge/React-18.2+-blue.svg)](https://reactjs.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.28+-red.svg)](https://github.com/google/jax)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-CausaLM-red.svg)](https://arxiv.org/abs/2404.causallm)

React + JAX framework that injects do-calculus interventions into UI prototypes to test LLM causal reasoning. Convert Figma designs into causal reasoning experiments.

## 🎯 Overview

Building on Stanford's CausaLM findings about weak causal modeling in LLMs, this framework uniquely combines:

- **Figma → Causal UI** converter for rapid experiment design
- **React components** with built-in intervention tracking
- **JAX backend** for efficient causal computation
- **Real-time metrics** (TE, ATE error) during user interaction
- **LLM agent baselines** for comparative analysis

## ✨ Key Differentiators

Unlike other causal toolkits, Causal UI Gym focuses on:
- **Visual UI-based** causal reasoning tests (not just code)
- **Designer-friendly** workflow starting from Figma
- **Production React** components you can embed anywhere
- **JAX-powered** backend for scalable causal computations
- **Automatic metric extraction** from UI interactions

## 📋 Requirements

```bash
# Frontend
react>=18.2.0
react-dom>=18.2.0
typescript>=5.0.0
@mui/material>=5.15.0
d3>=7.9.0
framer-motion>=11.0.0
recharts>=2.12.0

# Backend
jax>=0.4.28
jaxlib>=0.4.28
fastapi>=0.110.0
uvicorn>=0.30.0

# Causal & ML
pgmpy>=0.1.25
causalnex>=0.12.0
dowhy>=0.11.0
openai>=1.35.0
anthropic>=0.30.0

# Development
vite>=5.2.0
vitest>=1.6.0
playwright>=1.44.0
storybook>=8.1.0
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/causal-ui-gym.git
cd causal-ui-gym

# Install dependencies
npm install              # Frontend
pip install -r requirements.txt  # Backend

# Start development servers
npm run dev              # Frontend on :5173
python -m causal_ui_gym.server  # Backend on :8000
```

## 🚀 Quick Start

### 1. Convert Figma Design

```bash
# Export Figma design to causal components
npx causal-ui figma-import \
  --token YOUR_FIGMA_TOKEN \
  --file YOUR_FILE_ID \
  --output src/experiments/
```

### 2. Define Causal Structure

```typescript
// src/experiments/pricing-experiment.tsx
import { CausalExperiment, InterventionButton } from '@causal-ui/react'

export function PricingExperiment() {
  const causalModel = {
    nodes: ['price', 'demand', 'revenue', 'competitor_price'],
    edges: [
      ['price', 'demand'],
      ['price', 'revenue'],
      ['demand', 'revenue'],
      ['competitor_price', 'demand']
    ]
  }

  return (
    <CausalExperiment 
      model={causalModel}
      trackLLM="gpt-4"
      metrics={['ate_error', 'causal_accuracy']}
    >
      <div className="pricing-ui">
        <InterventionButton 
          variable="price" 
          value={29.99}
          label="Set Price to $29.99"
        />
        
        <ObservationPanel 
          variable="demand"
          visualization="line-chart" 
        />
        
        <CausalQuestion
          question="If we increase price by $10, what happens to revenue?"
          groundTruth={calculateATE}
        />
      </div>
    </CausalExperiment>
  )
}
```

### 3. Run Experiment with LLMs

```python
# backend/run_experiment.py
from causal_ui_gym import ExperimentRunner, LLMAgent

runner = ExperimentRunner()

# Test multiple LLMs
agents = [
    LLMAgent("gpt-4"),
    LLMAgent("claude-3"),
    LLMAgent("llama-3-70b")
]

results = runner.batch_evaluate(
    experiment_id="pricing-experiment",
    agents=agents,
    num_interventions=20,
    measure_beliefs=True
)

# Analyze causal reasoning
print(results.summary())
results.plot_ate_errors()
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Figma Plugin  │────▶│  Converter   │────▶│ React Component │
│                 │     │              │     │   + Causal DAG  │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                               ┌──────────────────────┘
                               ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   JAX Backend   │◀────│ Intervention │◀────│   User/LLM      │
│ (Do-Calculus)   │     │   Tracker    │     │   Interaction   │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## 🎨 React Components

### Core Components

```tsx
import { 
  CausalGraph,
  InterventionControl,
  BeliefTracker,
  MetricsDisplay 
} from '@causal-ui/react'

// Interactive causal graph with interventions
<CausalGraph
  nodes={nodes}
  edges={edges}
  onIntervene={(node, value) => handleIntervention(node, value)}
  highlightBackdoorPaths
  animateInterventions
/>

// Slider for continuous interventions
<InterventionControl
  variable="temperature"
  min={0}
  max={100}
  step={1}
  onIntervene={(value) => setTemperature(value)}
/>

// Real-time belief tracking
<BeliefTracker
  agent="gpt-4"
  beliefs={currentBeliefs}
  groundTruth={causalGroundTruth}
  showDivergence
/>
```

### Pre-built Experiments

```tsx
// Economics experiment
import { SupplyDemandExperiment } from '@causal-ui/experiments'

<SupplyDemandExperiment
  goods={['wheat', 'bread']}
  allowPriceControls
  trackMetrics
/>

// Medical diagnosis
import { SymptomDiseaseExperiment } from '@causal-ui/experiments'

<SymptomDiseaseExperiment
  diseases={['flu', 'covid', 'cold']}
  symptoms={['fever', 'cough', 'fatigue']}
  allowTreatmentInterventions
/>
```

## 🔧 JAX Backend

### Causal Computation Engine

```python
# causal_ui_gym/backend/engine.py
import jax
import jax.numpy as jnp
from causal_ui_gym.backend import CausalEngine

class JaxCausalEngine(CausalEngine):
    @jax.jit
    def compute_intervention(self, dag, intervention, evidence):
        """GPU-accelerated do-calculus"""
        # Mutilate graph for intervention
        mutilated_dag = self.mutilate(dag, intervention)
        
        # Compute intervention distribution
        return self.infer_jax(mutilated_dag, evidence)
    
    @jax.jit
    def compute_ate(self, dag, treatment, outcome, covariates):
        """Average Treatment Effect with JAX"""
        do_1 = self.compute_intervention(dag, {treatment: 1}, covariates)
        do_0 = self.compute_intervention(dag, {treatment: 0}, covariates)
        
        return jnp.mean(do_1[outcome] - do_0[outcome])
```

### Scalable Inference

```python
# Vectorized causal inference for multiple interventions
@jax.vmap
def batch_interventions(interventions, dag):
    return compute_intervention(dag, interventions)

# Parallel belief updates for multiple agents
@jax.pmap
def update_agent_beliefs(agent_states, observations):
    return jax.vmap(update_single_belief)(agent_states, observations)
```

## 📊 Metrics & Analysis

### Real-time Metrics Dashboard

```typescript
// Frontend metrics component
export function CausalMetricsDashboard({ experimentId }) {
  const { metrics, isLoading } = useCausalMetrics(experimentId)
  
  return (
    <Grid container spacing={2}>
      <Grid item xs={6}>
        <ATEErrorChart 
          data={metrics.ateErrors}
          agents={metrics.agents}
        />
      </Grid>
      
      <Grid item xs={6}>
        <CausalAccuracyHeatmap
          data={metrics.accuracyMatrix}
          variables={metrics.variables}
        />
      </Grid>
      
      <Grid item xs={12}>
        <InterventionTimeline
          interventions={metrics.interventionHistory}
          beliefs={metrics.beliefTrajectories}
        />
      </Grid>
    </Grid>
  )
}
```

### Automated Reporting

```python
from causal_ui_gym.analysis import ExperimentAnalyzer

analyzer = ExperimentAnalyzer()

# Generate comprehensive report
report = analyzer.analyze_experiment(
    experiment_id="pricing-ui-test",
    metrics=['ate_error', 'total_effect_error', 'backdoor_identification'],
    compare_agents=['gpt-4', 'claude-3', 'human-baseline']
)

# Export results
report.to_latex("results/causal_analysis.tex")
report.to_dashboard("http://localhost:3000/results")
```

## 🎭 Figma Integration

### Design-to-Experiment Workflow

1. **Design in Figma** with causal annotations
2. **Export** using our Figma plugin
3. **Auto-generate** React components
4. **Configure** causal relationships
5. **Deploy** experiment

### Figma Plugin

```typescript
// figma-plugin/code.ts
figma.ui.onmessage = msg => {
  if (msg.type === 'create-causal-annotation') {
    const selection = figma.currentPage.selection
    
    // Add causal metadata to components
    selection.forEach(node => {
      node.setPluginData('causalVariable', msg.variable)
      node.setPluginData('interventionType', msg.type)
    })
  }
  
  if (msg.type === 'export-to-causal-ui') {
    const causalComponents = extractCausalComponents()
    figma.ui.postMessage({
      type: 'export-complete',
      components: causalComponents
    })
  }
}
```

## 🧪 Testing Framework

### Visual Regression Tests

```typescript
// tests/visual-regression.spec.ts
import { test, expect } from '@playwright/test'

test('intervention changes visualization correctly', async ({ page }) => {
  await page.goto('/experiments/supply-demand')
  
  // Baseline screenshot
  await expect(page).toHaveScreenshot('baseline.png')
  
  // Perform intervention
  await page.click('[data-intervention="price"]')
  await page.fill('[data-intervention-value]', '150')
  
  // Check visual changes
  await expect(page).toHaveScreenshot('after-intervention.png')
  
  // Verify causal metrics updated
  const ateError = await page.textContent('[data-metric="ate-error"]')
  expect(parseFloat(ateError)).toBeLessThan(0.1)
})
```

### Causal Reasoning Tests

```python
# tests/test_causal_reasoning.py
def test_backdoor_identification():
    """Test if LLMs identify backdoor paths correctly"""
    dag = create_confounded_dag()
    
    llm_response = agent.identify_backdoors(dag, 'X', 'Y')
    ground_truth = compute_backdoor_paths(dag, 'X', 'Y')
    
    assert set(llm_response) == set(ground_truth)
```

## 🚀 Deployment

### Vercel Deployment

```bash
# Deploy frontend
npm run build
vercel --prod

# Deploy JAX backend on Modal
modal deploy causal_ui_gym.backend.app
```

### Docker Compose

```yaml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://backend:8000
      
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- New experiment templates
- Additional causal metrics
- Figma plugin improvements
- LLM agent implementations
- Documentation and tutorials

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

```bibtex
@software{causal_ui_gym,
  title={Causal UI Gym: Visual Framework for Testing LLM Causal Reasoning},
  author={Daniel Schmidt},
  year={2025},
  url={https://github.com/danieleschmidt/causal-ui-gym}
}
```

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 🔗 Resources

- [Documentation](https://causal-ui-gym.dev)
- [Storybook Components](https://storybook.causal-ui-gym.dev)
- [Figma Plugin](https://www.figma.com/community/plugin/causal-ui-gym)
- [Example Experiments](https://github.com/causal-ui-gym/examples)
- [Discord Community](https://discord.gg/causal-ui)

## 📧 Contact

- **GitHub Issues**: Bug reports and features
- **Email**: causal-ui@yourdomain.com
- **Twitter**: [@CausalUIGym](https://twitter.com/causaluigym)
