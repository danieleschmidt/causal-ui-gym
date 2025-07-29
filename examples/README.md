# Examples

This directory contains example experiments demonstrating how to use Causal UI Gym.

## Available Examples

### Basic Experiment (`basic-experiment.tsx`)

A simple supply and demand model that demonstrates:
- Basic causal graph visualization
- Node-based interventions
- Intervention tracking

**Usage:**
```typescript
import { BasicExperiment } from './examples/basic-experiment'

function App() {
  return <BasicExperiment />
}
```

## Running Examples

1. Start the development server:
   ```bash
   npm run dev
   ```

2. Import and use the example components in your application

3. Customize the causal models and experiment parameters as needed

## Creating New Examples

1. Create a new `.tsx` file in this directory
2. Define your causal DAG structure
3. Use the provided components to build your experiment UI
4. Add documentation to this README

## Example Structure

Each example should follow this pattern:

```typescript
import React from 'react'
import { CausalGraph } from '../src/components'
import { CausalDAG } from '../src/types'

// Define your causal model
const MY_DAG: CausalDAG = {
  nodes: [/* ... */],
  edges: [/* ... */]
}

export function MyExperiment() {
  // Component implementation
  return (
    <div>
      <CausalGraph dag={MY_DAG} onIntervene={handleIntervention} />
      {/* Additional UI components */}
    </div>
  )
}
```

## Contributing Examples

We welcome contributions of new examples! Please:

1. Follow the existing code style
2. Include clear documentation
3. Test your example thoroughly
4. Submit a pull request with your changes