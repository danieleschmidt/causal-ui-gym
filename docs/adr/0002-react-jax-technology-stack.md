# ADR-0002: React + JAX Technology Stack

## Status
Accepted

## Context
We need to select appropriate technologies for building a framework that tests LLM causal reasoning through interactive UI components. The system requires:
- Interactive frontend for causal graph visualization
- High-performance backend for causal computations
- Real-time updates during experiments
- Integration with multiple LLM providers

## Decision
We will use:
- **Frontend**: React 18+ with TypeScript for UI components
- **Backend**: JAX with Python for causal computations  
- **Visualization**: D3.js for interactive graph rendering
- **Styling**: Material-UI for consistent design system
- **Real-time**: WebSockets for live experiment updates

## Alternatives Considered
1. **Vue.js + PyTorch**: Less TypeScript ecosystem, slower numerical computing
2. **Svelte + TensorFlow**: Smaller community, more complex deployment
3. **Angular + NumPy**: More complex framework, insufficient GPU acceleration

## Consequences
**Positive:**
- React provides mature ecosystem and component reusability
- JAX enables GPU-accelerated causal inference with JIT compilation
- TypeScript ensures type safety across complex causal models
- D3.js allows sophisticated graph visualizations
- Strong community support for all technologies

**Negative:**
- JAX has a steeper learning curve compared to NumPy
- React bundle size may be larger than alternatives
- Requires expertise in both frontend and scientific computing