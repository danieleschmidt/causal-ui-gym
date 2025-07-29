import React, { useState } from 'react'
import { CausalGraph } from '../src/components'
import { CausalDAG } from '../src/types'

// Example: Simple supply and demand causal model
const SUPPLY_DEMAND_DAG: CausalDAG = {
  nodes: [
    { id: 'price', label: 'Price', position: { x: 200, y: 50 } },
    { id: 'supply', label: 'Supply', position: { x: 100, y: 150 } },
    { id: 'demand', label: 'Demand', position: { x: 300, y: 150 } },
    { id: 'quantity', label: 'Quantity', position: { x: 200, y: 250 } }
  ],
  edges: [
    { source: 'price', target: 'supply' },
    { source: 'price', target: 'demand' },
    { source: 'supply', target: 'quantity' },
    { source: 'demand', target: 'quantity' }
  ]
}

export function BasicExperiment() {
  const [interventions, setInterventions] = useState<Array<{nodeId: string, value: number}>>([])
  
  const handleIntervention = (nodeId: string, value: number) => {
    setInterventions(prev => [...prev, { nodeId, value }])
    console.log(`Intervention: Set ${nodeId} to ${value}`)
  }
  
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Supply & Demand Causal Model</h1>
      
      <div className="mb-6">
        <h2 className="text-lg font-semibold mb-2">Causal Graph</h2>
        <p className="text-gray-600 mb-4">
          Click on nodes to perform interventions. This model shows how price
          affects both supply and demand, which in turn determine quantity sold.
        </p>
        
        <CausalGraph 
          dag={SUPPLY_DEMAND_DAG} 
          onIntervene={handleIntervention}
          className="border rounded p-4 bg-white"
        />
      </div>
      
      <div>
        <h2 className="text-lg font-semibold mb-2">Intervention History</h2>
        {interventions.length === 0 ? (
          <p className="text-gray-500">No interventions performed yet.</p>
        ) : (
          <ul className="space-y-1">
            {interventions.map((intervention, index) => (
              <li key={index} className="text-sm">
                Set <strong>{intervention.nodeId}</strong> to {intervention.value}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}