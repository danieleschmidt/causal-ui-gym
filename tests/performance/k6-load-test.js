import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up to 100 users
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp up to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(99)<1500'], // 99% of requests under 1.5s
    http_req_failed: ['rate<0.1'],     // Error rate under 10%
    errors: ['rate<0.1'],              // Custom error rate under 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // Test causal graph computation endpoint
  const causalGraphPayload = {
    nodes: ['price', 'demand', 'revenue'],
    edges: [['price', 'demand'], ['price', 'revenue'], ['demand', 'revenue']],
    intervention: { price: 29.99 }
  };

  const response = http.post(`${BASE_URL}/api/causal/compute`, 
    JSON.stringify(causalGraphPayload), {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Check response
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
    'has causal result': (r) => JSON.parse(r.body).result !== undefined,
  });

  errorRate.add(!success);
  sleep(1);
}

// Spike test configuration
export function spikeTest() {
  return {
    stages: [
      { duration: '10s', target: 100 },  // Below normal load
      { duration: '1m', target: 100 },
      { duration: '10s', target: 1400 }, // Spike to 1400 users
      { duration: '3m', target: 1400 },  // Stay at 1400 for 3 minutes
      { duration: '10s', target: 100 },  // Scale down
      { duration: '3m', target: 100 },
      { duration: '10s', target: 0 },
    ],
  };
}