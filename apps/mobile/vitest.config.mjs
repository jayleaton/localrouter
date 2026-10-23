import path from 'path'
import { fileURLToPath } from 'url'
import { defineConfig } from 'vitest/config'

const dirname = path.dirname(fileURLToPath(import.meta.url))

// The pure parts only (protocol, conversation, voice decisions); native code is checked on a device.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
  resolve: { alias: { '@': path.join(dirname, 'src') } },
})
