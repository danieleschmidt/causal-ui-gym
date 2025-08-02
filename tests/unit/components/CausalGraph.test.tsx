/**
 * Unit tests for CausalGraph component
 */

import { screen, fireEvent } from '@testing-library/react'
import { render, mockCausalModel } from '../../utils/test-helpers'
import { CausalGraph } from '../../../src/components/CausalGraph'

describe('CausalGraph Component', () => {
  it('renders all nodes correctly', () => {
    render(<CausalGraph nodes={mockCausalModel.nodes} edges={mockCausalModel.edges} />)
    
    expect(screen.getByText('Price')).toBeInTheDocument()
    expect(screen.getByText('Demand')).toBeInTheDocument()
    expect(screen.getByText('Revenue')).toBeInTheDocument()
  })

  it('renders edges between nodes', () => {
    render(<CausalGraph nodes={mockCausalModel.nodes} edges={mockCausalModel.edges} />)
    
    // Check that edges are rendered (this would depend on the actual implementation)
    const svgElement = screen.getByRole('img', { hidden: true })
    expect(svgElement).toBeInTheDocument()
  })

  it('handles node click events', () => {
    const mockOnNodeClick = vi.fn()
    render(
      <CausalGraph 
        nodes={mockCausalModel.nodes} 
        edges={mockCausalModel.edges}
        onNodeClick={mockOnNodeClick}
      />
    )
    
    const priceNode = screen.getByText('Price')
    fireEvent.click(priceNode)
    
    expect(mockOnNodeClick).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'price' })
    )
  })

  it('highlights interventions correctly', () => {
    const interventions = [{ variable: 'price', value: 50 }]
    render(
      <CausalGraph 
        nodes={mockCausalModel.nodes} 
        edges={mockCausalModel.edges}
        interventions={interventions}
      />
    )
    
    const priceNode = screen.getByText('Price')
    expect(priceNode).toHaveClass('intervened') // Assuming this class exists
  })

  it('displays node tooltips on hover', async () => {
    render(<CausalGraph nodes={mockCausalModel.nodes} edges={mockCausalModel.edges} />)
    
    const priceNode = screen.getByText('Price')
    fireEvent.mouseEnter(priceNode)
    
    // Check for tooltip content (implementation dependent)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
  })

  it('handles empty data gracefully', () => {
    render(<CausalGraph nodes={[]} edges={[]} />)
    
    expect(screen.getByText(/no data/i)).toBeInTheDocument()
  })

  it('supports keyboard navigation', () => {
    render(<CausalGraph nodes={mockCausalModel.nodes} edges={mockCausalModel.edges} />)
    
    const firstNode = screen.getByText('Price')
    firstNode.focus()
    
    expect(document.activeElement).toBe(firstNode)
    
    // Test tab navigation
    fireEvent.keyDown(firstNode, { key: 'Tab' })
    expect(document.activeElement).not.toBe(firstNode)
  })

  it('renders with different layouts', () => {
    const layouts = ['force', 'hierarchy', 'circular']
    
    layouts.forEach(layout => {
      const { container } = render(
        <CausalGraph 
          nodes={mockCausalModel.nodes} 
          edges={mockCausalModel.edges}
          layout={layout}
        />
      )
      
      expect(container.firstChild).toHaveAttribute('data-layout', layout)
    })
  })

  it('updates when data changes', () => {
    const { rerender } = render(
      <CausalGraph nodes={mockCausalModel.nodes} edges={mockCausalModel.edges} />
    )
    
    expect(screen.getByText('Price')).toBeInTheDocument()
    
    const newNodes = [{ id: 'temp', label: 'Temperature', type: 'continuous' as const }]
    rerender(<CausalGraph nodes={newNodes} edges={[]} />)
    
    expect(screen.queryByText('Price')).not.toBeInTheDocument()
    expect(screen.getByText('Temperature')).toBeInTheDocument()
  })

  it('handles large graphs efficiently', () => {
    const largeNodes = Array.from({ length: 100 }, (_, i) => ({
      id: `node_${i}`,
      label: `Node ${i}`,
      type: 'continuous' as const,
    }))
    
    const largeEdges = Array.from({ length: 99 }, (_, i) => ({
      from: `node_${i}`,
      to: `node_${i + 1}`,
      relationship: 'positive' as const,
    }))
    
    const startTime = performance.now()
    render(<CausalGraph nodes={largeNodes} edges={largeEdges} />)
    const renderTime = performance.now() - startTime
    
    // Ensure rendering is reasonably fast (less than 1 second)
    expect(renderTime).toBeLessThan(1000)
  })

  it('displays error state for invalid data', () => {
    const invalidNodes = [{ id: '', label: '', type: 'invalid' as any }]
    
    render(<CausalGraph nodes={invalidNodes} edges={[]} />)
    
    expect(screen.getByText(/error/i)).toBeInTheDocument()
  })

  it('supports zooming and panning', () => {
    const { container } = render(
      <CausalGraph 
        nodes={mockCausalModel.nodes} 
        edges={mockCausalModel.edges}
        enableZoom={true}
      />
    )
    
    const svgElement = container.querySelector('svg')
    expect(svgElement).toHaveAttribute('data-zoom-enabled', 'true')
    
    // Test zoom event
    fireEvent.wheel(svgElement!, { deltaY: -100 })
    
    // Check that zoom transform is applied (implementation dependent)
    const zoomGroup = svgElement!.querySelector('g[data-zoom-group]')
    expect(zoomGroup).toHaveAttribute('transform')
  })
})