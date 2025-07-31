// Lighthouse CI configuration for performance monitoring
module.exports = {
  ci: {
    collect: {
      url: ['http://localhost:5173', 'http://localhost:5173/examples'],
      startServerCommand: 'npm run dev',
      startServerReadyPattern: 'Local:.*http://localhost:5173',
      startServerReadyTimeout: 30000,
      numberOfRuns: 3,
      settings: {
        preset: 'desktop',
        onlyCategories: ['performance', 'accessibility', 'best-practices'],
        skipAudits: ['uses-http2'],
      },
    },
    assert: {
      assertions: {
        'categories:performance': ['error', {minScore: 0.8}],
        'categories:accessibility': ['error', {minScore: 0.9}],
        'categories:best-practices': ['error', {minScore: 0.8}],
        'first-contentful-paint': ['error', {maxNumericValue: 2000}],
        'largest-contentful-paint': ['error', {maxNumericValue: 3000}],
        'cumulative-layout-shift': ['error', {maxNumericValue: 0.1}],
        'total-blocking-time': ['error', {maxNumericValue: 300}],
      },
    },
    upload: {
      target: 'temporary-public-storage',
    },
  },
};