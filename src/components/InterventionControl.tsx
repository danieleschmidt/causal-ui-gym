import React, { useState, useCallback } from 'react'
import { Slider, TextField, Button, Typography, Box, Paper, Chip } from '@mui/material'
import { Intervention } from '../types'
import { debounce } from '../utils'

interface InterventionControlProps {
  variable: string
  currentValue?: number
  min?: number
  max?: number
  step?: number
  onIntervene: (intervention: Intervention) => void
  disabled?: boolean
  className?: string
}

export function InterventionControl({
  variable,
  currentValue = 0,
  min = 0,
  max = 100,
  step = 1,
  onIntervene,
  disabled = false,
  className
}: InterventionControlProps) {
  const [value, setValue] = useState(currentValue)
  const [isInterventionActive, setIsInterventionActive] = useState(false)

  const debouncedIntervene = useCallback(
    debounce((newValue: number) => {
      const intervention: Intervention = {
        variable,
        value: newValue,
        intervention_type: 'do',
        timestamp: new Date()
      }
      onIntervene(intervention)
    }, 300),
    [variable, onIntervene]
  )

  const handleSliderChange = (_: Event, newValue: number | number[]) => {
    const numericValue = Array.isArray(newValue) ? newValue[0] : newValue
    setValue(numericValue)
    debouncedIntervene(numericValue)
  }

  const handleTextChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(event.target.value) || 0
    setValue(newValue)
    debouncedIntervene(newValue)
  }

  const handleActivateIntervention = () => {
    setIsInterventionActive(true)
    const intervention: Intervention = {
      variable,
      value,
      intervention_type: 'do',
      timestamp: new Date()
    }
    onIntervene(intervention)
  }

  const handleResetIntervention = () => {
    setIsInterventionActive(false)
    setValue(currentValue)
    const intervention: Intervention = {
      variable,
      value: currentValue,
      intervention_type: 'do',
      timestamp: new Date()
    }
    onIntervene(intervention)
  }

  return (
    <Paper className={`intervention-control ${className || ''}`} sx={{ p: 2, mb: 2 }}>
      <Box display="flex" alignItems="center" mb={1}>
        <Typography variant="h6" component="h3" sx={{ flexGrow: 1 }}>
          Intervention on {variable}
        </Typography>
        {isInterventionActive && (
          <Chip 
            label="Active" 
            color="primary" 
            size="small"
            onDelete={handleResetIntervention}
          />
        )}
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Current Value: {value.toFixed(2)}
        </Typography>
        
        <Slider
          value={value}
          onChange={handleSliderChange}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          valueLabelDisplay="auto"
          sx={{ mb: 2 }}
        />
      </Box>

      <Box display="flex" gap={2} alignItems="center">
        <TextField
          type="number"
          value={value}
          onChange={handleTextChange}
          disabled={disabled}
          size="small"
          inputProps={{ min, max, step }}
          sx={{ width: '120px' }}
        />
        
        <Button
          variant={isInterventionActive ? "outlined" : "contained"}
          onClick={isInterventionActive ? handleResetIntervention : handleActivateIntervention}
          disabled={disabled}
        >
          {isInterventionActive ? 'Reset' : 'Intervene'}
        </Button>
      </Box>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        Range: {min} to {max} (step: {step})
      </Typography>
    </Paper>
  )
}