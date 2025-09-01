const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const compression = require('compression');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet());
app.use(cors());
app.use(compression());
app.use(express.json({ limit: '10mb' }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
});
app.use('/api/', limiter);

// In-memory data store (replace with database in production)
const dataStore = new Map();
const sessions = new Map();

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Data processing endpoint
app.post('/api/process', async (req, res) => {
  try {
    const { data, options = {} } = req.body;
    
    if (!data) {
      return res.status(400).json({ error: 'Data required' });
    }
    
    // Simulate processing
    const startTime = Date.now();
    const processed = await processData(data, options);
    const processingTime = Date.now() - startTime;
    
    // Store result
    const resultId = generateId();
    dataStore.set(resultId, {
      input: data,
      output: processed,
      timestamp: new Date(),
      processingTime
    });
    
    res.json({
      id: resultId,
      result: processed,
      processingTime,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Processing error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get processing history
app.get('/api/history', (req, res) => {
  const { limit = 10, offset = 0 } = req.query;
  const results = Array.from(dataStore.entries())
    .slice(offset, offset + limit)
    .map(([id, data]) => ({ id, ...data }));
  
  res.json({
    results,
    total: dataStore.size,
    limit: parseInt(limit),
    offset: parseInt(offset)
  });
});

// Get specific result
app.get('/api/result/:id', (req, res) => {
  const result = dataStore.get(req.params.id);
  if (!result) {
    return res.status(404).json({ error: 'Result not found' });
  }
  res.json(result);
});

// WebSocket support for real-time updates
const WebSocket = require('ws');
const server = require('http').createServer(app);
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  const sessionId = generateId();
  sessions.set(sessionId, ws);
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      handleWebSocketMessage(sessionId, data);
    } catch (error) {
      ws.send(JSON.stringify({ error: error.message }));
    }
  });
  
  ws.on('close', () => {
    sessions.delete(sessionId);
  });
  
  ws.send(JSON.stringify({ 
    type: 'connected', 
    sessionId,
    timestamp: new Date().toISOString()
  }));
});

// Helper functions
async function processData(data, options) {
  // Simulate complex processing
  await new Promise(resolve => setTimeout(resolve, Math.random() * 1000));
  
  if (Array.isArray(data)) {
    return data.map(item => ({
      ...item,
      processed: true,
      score: Math.random(),
      tags: generateTags(item)
    }));
  }
  
  return {
    ...data,
    processed: true,
    analysis: {
      complexity: JSON.stringify(data).length,
      fields: Object.keys(data).length,
      timestamp: new Date().toISOString()
    }
  };
}

function generateTags(item) {
  const tags = [];
  if (item.priority > 0.7) tags.push('high-priority');
  if (item.value > 100) tags.push('high-value');
  return tags;
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function handleWebSocketMessage(sessionId, data) {
  const ws = sessions.get(sessionId);
  if (!ws) return;
  
  switch (data.type) {
    case 'subscribe':
      // Handle subscription to updates
      ws.send(JSON.stringify({ type: 'subscribed', channel: data.channel }));
      break;
    case 'process':
      // Handle real-time processing
      processData(data.payload).then(result => {
        ws.send(JSON.stringify({ type: 'result', data: result }));
      });
      break;
  }
}

// Start server
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
