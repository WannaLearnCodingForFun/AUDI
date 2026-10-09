import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API address must match `api.host` / `api.port` in config.yaml; override with AUDI_API_URL.
const api = process.env.AUDI_API_URL || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: api, changeOrigin: false },
      '/ws': { target: api.replace(/^http/, 'ws'), ws: true },
    },
  },
})
