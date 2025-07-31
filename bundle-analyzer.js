// Bundle size analysis for performance optimization
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const { defineConfig } = require('vite');

// Bundle size thresholds (in KB)
const BUNDLE_SIZE_LIMITS = {
  // Main application bundle
  'index': 500,
  // Vendor libraries (React, D3, etc.)
  'vendor': 800,
  // JAX bridge and causal computation
  'causal': 300,
  // UI components and visualization
  'components': 400,
};

// Generate bundle analysis report
function analyzeBundleSize() {
  const analysis = {
    timestamp: new Date().toISOString(),
    limits: BUNDLE_SIZE_LIMITS,
    recommendations: [
      'Consider code splitting for causal computation modules',
      'Lazy load D3 visualizations only when needed',
      'Use dynamic imports for ML model loading',
      'Implement route-based code splitting',
    ],
  };
  
  console.log('Bundle Analysis Configuration:', JSON.stringify(analysis, null, 2));
  return analysis;
}

// Vite plugin for bundle analysis
export const bundleAnalyzerConfig = defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          ui: ['@mui/material', 'framer-motion'],
          viz: ['d3', 'recharts'],
          causal: ['jax'], // If JAX client exists
        },
      },
    },
  },
  plugins: [
    // Add to vite.config.ts for production builds
    // BundleAnalyzerPlugin({ analyzerMode: 'static' })
  ],
});

module.exports = { analyzeBundleSize, bundleAnalyzerConfig, BUNDLE_SIZE_LIMITS };