# Getting Started with Causal UI Gym

Welcome to Causal UI Gym! This guide will help you create your first causal reasoning experiment in under 10 minutes.

## 🎯 What You'll Learn

By the end of this guide, you'll have:
- Set up the development environment
- Created a simple pricing experiment
- Tested LLM causal reasoning capabilities
- Understood the core concepts and workflow

## 📋 Prerequisites

Before starting, ensure you have:
- Node.js 18+ installed
- Python 3.9+ installed
- Basic understanding of React concepts
- Familiarity with causal inference (helpful but not required)

## 🚀 Quick Start

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/causal-ui-gym/causal-ui-gym.git
cd causal-ui-gym

# Install frontend dependencies
npm install

# Install backend dependencies
pip install -r requirements.txt
```

### Step 2: Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Add your API keys (optional for local development)
echo "OPENAI_API_KEY=your-key-here" >> .env
echo "ANTHROPIC_API_KEY=your-key-here" >> .env
```

### Step 3: Start Development Servers

```bash
# Terminal 1: Start frontend
npm run dev

# Terminal 2: Start backend
python -m causal_ui_gym.server
```

Visit http://localhost:5173 to see the application running!

## 🧪 Your First Experiment

Let's create a simple pricing experiment to test how LLMs understand the relationship between price, demand, and revenue.

### Define the Causal Model

Create `src/experiments/my-first-experiment.tsx`:

```tsx
import React from 'react'
import { CausalExperiment, CausalGraph, InterventionButton } from '@causal-ui/react'

export function MyFirstExperiment() {
  // Define causal relationships
  const causalModel = {
    nodes: [
      { id: 'price', label: 'Price ($)', type: 'continuous' },
      { id: 'demand', label: 'Demand', type: 'continuous' },
      { id: 'revenue', label: 'Revenue ($)', type: 'continuous' }
    ],
    edges: [
      { from: 'price', to: 'demand', relationship: 'negative' },
      { from: 'price', to: 'revenue', relationship: 'positive' },
      { from: 'demand', to: 'revenue', relationship: 'positive' }
    ]
  }

  // Ground truth for validation
  const computeRevenue = (price: number, demand: number) => {
    return price * demand
  }

  const computeDemand = (price: number) => {
    return Math.max(0, 100 - price * 0.8) // Linear demand curve
  }

  return (
    <CausalExperiment
      model={causalModel}
      title="Pricing Strategy Experiment"
      description="Test how price changes affect demand and revenue"
    >
      <div className="experiment-layout">
        {/* Interactive causal graph */}
        <CausalGraph
          model={causalModel}
          onNodeClick={(node) => console.log('Selected:', node)}
          showMetrics={true}
        />
        
        {/* Intervention controls */}
        <div className="intervention-panel">
          <h3>Test Interventions</h3>
          
          <InterventionButton
            variable="price"
            value={29.99}
            label="Set Price to $29.99"
            color="primary"
          />
          
          <InterventionButton
            variable="price"
            value={49.99}
            label="Set Price to $49.99"
            color="secondary"
          />
          
          <InterventionButton
            variable="price"
            value={99.99}
            label="Set Price to $99.99"
            color="warning"
          />
        </div>
        
        {/* Results display */}
        <div className="results-panel">
          <h3>Causal Question</h3>
          <p>
            "If we increase the price from $30 to $50, 
            what happens to total revenue?"
          </p>
          
          <div className="llm-predictions">
            {/* LLM responses will appear here */}
          </div>
        </div>
      </div>
    </CausalExperiment>
  )
}
```

### Add to Main App

Update `src/App.tsx`:

```tsx
import { MyFirstExperiment } from './experiments/my-first-experiment'

function App() {
  return (
    <div className="App">
      <header>
        <h1>Causal UI Gym</h1>
        <p>Testing LLM Causal Reasoning</p>
      </header>
      
      <main>
        <MyFirstExperiment />
      </main>
    </div>
  )
}

export default App
```

### Test with LLMs

```typescript
// In your experiment component
import { useLLMAgent } from '@causal-ui/hooks'

export function MyFirstExperiment() {
  const { queryAgent, results, isLoading } = useLLMAgent({
    model: 'gpt-4',
    systemPrompt: 'You are analyzing causal relationships in a pricing scenario.'
  })

  const handleIntervention = async (variable: string, value: number) => {
    const question = `If ${variable} is set to ${value}, what happens to revenue?`
    
    await queryAgent({
      question,
      context: causalModel,
      intervention: { [variable]: value }
    })
  }

  // ... rest of component
}
```

## 📊 Understanding the Results

After running your experiment, you'll see:

### 1. Causal Graph Visualization
- Nodes represent variables (price, demand, revenue)
- Edges show causal relationships
- Colors indicate intervention effects

### 2. LLM Predictions
- Agent responses to causal questions
- Confidence scores for predictions
- Comparison with ground truth

### 3. Metrics Dashboard
- **ATE Error**: How far off the LLM's estimate is
- **Causal Accuracy**: Percentage of correct causal relationships identified
- **Intervention Success**: Whether interventions produce expected results

## 🔍 Key Concepts

### Causal DAG (Directed Acyclic Graph)
- **Nodes**: Variables in your system
- **Edges**: Direct causal relationships
- **Interventions**: Setting variables to specific values

### Do-Calculus
- Mathematical framework for causal reasoning
- Enables computation of intervention effects
- Implemented efficiently using JAX

### LLM Evaluation
- Compare LLM predictions with ground truth
- Measure causal reasoning accuracy
- Identify systematic biases

## 🎯 Next Steps

Now that you've created your first experiment:

1. **Explore Components**: Try different visualization options
2. **Add Complexity**: Include confounding variables
3. **Test Multiple LLMs**: Compare GPT-4, Claude, and local models
4. **Create Templates**: Build reusable experiment patterns
5. **Read Advanced Guides**: Dive deeper into specific topics

## 📚 Related Guides

- [Creating Experiments](./creating-experiments.md) - Build more complex scenarios
- [Component Library](./component-library.md) - Learn all available components
- [LLM Integration](./llm-integration.md) - Work with different models
- [Metrics and Analysis](./metrics-analysis.md) - Understand evaluation metrics

## 🐛 Troubleshooting

### Common Issues

**Frontend won't start:**
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Backend connection errors:**
```bash
# Check if backend is running
curl http://localhost:8000/health

# Restart backend with verbose logging
python -m causal_ui_gym.server --debug
```

**LLM API errors:**
- Verify API keys in `.env` file
- Check API rate limits and billing
- Use local models for development

### Getting Help

- 💬 [Discord Community](https://discord.gg/causal-ui)
- 🐛 [GitHub Issues](https://github.com/causal-ui-gym/issues)
- 📧 [Email Support](mailto:support@causal-ui-gym.dev)

## 🎉 Congratulations!

You've successfully created your first causal reasoning experiment! You're now ready to explore more advanced features and contribute to the growing community of researchers testing LLM causal capabilities.

---

*Need help? Join our [Discord community](https://discord.gg/causal-ui) for real-time support!*