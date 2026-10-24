import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
    // Small shared machines: two workers are plenty for these suites.
    maxWorkers: 2,
    minWorkers: 1,
  },
})
