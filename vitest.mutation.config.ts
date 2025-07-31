/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Mutation testing configuration for Vitest
 * Tests the quality of test suites by introducing code mutations
 */
export default defineConfig({
  plugins: [react()],
  test: {
    // Mutation testing specific configuration
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    
    // Coverage configuration for mutation testing
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      reportsDirectory: './coverage/mutation',
      
      // High coverage thresholds for mutation testing
      thresholds: {
        global: {
          branches: 90,
          functions: 90,
          lines: 90,
          statements: 90,
        },
        // Specific thresholds for causal computation modules
        './src/utils/causal/*.ts': {
          branches: 95,
          functions: 95,
          lines: 95,
          statements: 95,
        },
      },
      
      // Include patterns
      include: [
        'src/**/*.{ts,tsx}',
        '!src/**/*.d.ts',
        '!src/test/**',
      ],
      
      // Exclude test files and mocks
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.test.{ts,tsx}',
        '**/*.spec.{ts,tsx}',
        '**/mocks/**',
      ],
    },
    
    // Test execution configuration
    testTimeout: 30000, // Longer timeout for mutation tests
    
    // Mutation testing patterns
    include: [
      'src/**/*.{test,spec}.{ts,tsx}',
    ],
    
    // Mock configuration
    deps: {
      inline: ['@testing-library/jest-dom'],
    },
  },
  
  // Build configuration for testing
  esbuild: {
    target: 'node14',
  },
})

/*
 * Usage Instructions:
 * 
 * 1. Install mutation testing tools:
 *    npm install --save-dev @stryker-mutator/core @stryker-mutator/vitest-runner
 * 
 * 2. Run mutation tests:
 *    npx stryker run --configFile stryker.config.json
 * 
 * 3. View mutation testing report:
 *    open reports/mutation/html/index.html
 */