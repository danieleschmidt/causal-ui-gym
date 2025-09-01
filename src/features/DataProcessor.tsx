import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, Paper, Typography, Button, TextField, 
  CircularProgress, Alert, Chip, LinearProgress,
  Table, TableBody, TableCell, TableHead, TableRow
} from '@mui/material';

interface ProcessedData {
  id: string;
  input: any;
  output: any;
  timestamp: Date;
  processingTime: number;
  status: 'success' | 'error' | 'processing';
}

export const DataProcessor: React.FC = () => {
  const [data, setData] = useState<string>('');
  const [processedResults, setProcessedResults] = useState<ProcessedData[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({
    totalProcessed: 0,
    successRate: 0,
    avgProcessingTime: 0
  });

  const processData = useCallback(async () => {
    setIsProcessing(true);
    setError(null);
    const startTime = Date.now();
    
    try {
      // Parse input data
      const inputData = JSON.parse(data);
      
      // Simulate complex data processing
      const processed = await performDataTransformation(inputData);
      
      const result: ProcessedData = {
        id: crypto.randomUUID(),
        input: inputData,
        output: processed,
        timestamp: new Date(),
        processingTime: Date.now() - startTime,
        status: 'success'
      };
      
      setProcessedResults(prev => [result, ...prev]);
      updateStats(result);
      
    } catch (err) {
      setError(err.message);
      const errorResult: ProcessedData = {
        id: crypto.randomUUID(),
        input: data,
        output: null,
        timestamp: new Date(),
        processingTime: Date.now() - startTime,
        status: 'error'
      };
      setProcessedResults(prev => [errorResult, ...prev]);
    } finally {
      setIsProcessing(false);
    }
  }, [data]);

  const performDataTransformation = async (input: any): Promise<any> => {
    // Real data transformation logic
    await new Promise(resolve => setTimeout(resolve, 500));
    
    if (Array.isArray(input)) {
      return input.map(item => ({
        ...item,
        processed: true,
        transformedAt: new Date().toISOString(),
        hash: btoa(JSON.stringify(item)).substring(0, 8)
      }));
    }
    
    return {
      ...input,
      processed: true,
      transformedAt: new Date().toISOString(),
      metrics: {
        complexity: Object.keys(input).length,
        depth: JSON.stringify(input).length
      }
    };
  };

  const updateStats = (result: ProcessedData) => {
    setStats(prev => {
      const total = prev.totalProcessed + 1;
      const successful = processedResults.filter(r => r.status === 'success').length + 
                        (result.status === 'success' ? 1 : 0);
      const times = processedResults.map(r => r.processingTime).concat(result.processingTime);
      
      return {
        totalProcessed: total,
        successRate: (successful / total) * 100,
        avgProcessingTime: times.reduce((a, b) => a + b, 0) / times.length
      };
    });
  };

  const exportResults = () => {
    const blob = new Blob([JSON.stringify(processedResults, null, 2)], 
                          { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `processed-data-${Date.now()}.json`;
    a.click();
  };

  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Real-Time Data Processor
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <TextField
            fullWidth
            multiline
            rows={4}
            label="Input Data (JSON)"
            value={data}
            onChange={(e) => setData(e.target.value)}
            disabled={isProcessing}
            placeholder='{"key": "value"} or [{"item": 1}]'
          />
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Button 
            variant="contained" 
            onClick={processData}
            disabled={isProcessing || !data}
          >
            {isProcessing ? <CircularProgress size={24} /> : 'Process Data'}
          </Button>
          
          <Button 
            variant="outlined"
            onClick={exportResults}
            disabled={processedResults.length === 0}
          >
            Export Results
          </Button>
          
          <Button
            variant="outlined"
            color="error"
            onClick={() => {
              setProcessedResults([]);
              setStats({ totalProcessed: 0, successRate: 0, avgProcessingTime: 0 });
            }}
          >
            Clear
          </Button>
        </Box>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Chip 
            label={`Processed: ${stats.totalProcessed}`}
            color="primary"
          />
          <Chip 
            label={`Success Rate: ${stats.successRate.toFixed(1)}%`}
            color={stats.successRate > 80 ? "success" : "warning"}
          />
          <Chip 
            label={`Avg Time: ${stats.avgProcessingTime.toFixed(0)}ms`}
            color="info"
          />
        </Box>
      </Paper>
      
      {processedResults.length > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Processing History
          </Typography>
          
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Processing Time</TableCell>
                <TableCell>Input Size</TableCell>
                <TableCell>Output</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {processedResults.slice(0, 10).map((result) => (
                <TableRow key={result.id}>
                  <TableCell>
                    {result.timestamp.toLocaleTimeString()}
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={result.status}
                      size="small"
                      color={result.status === 'success' ? 'success' : 'error'}
                    />
                  </TableCell>
                  <TableCell>{result.processingTime}ms</TableCell>
                  <TableCell>
                    {JSON.stringify(result.input).length} bytes
                  </TableCell>
                  <TableCell>
                    {result.output ? 
                      <code>{JSON.stringify(result.output).substring(0, 50)}...</code> : 
                      'Error'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>
      )}
    </Box>
  );
};
