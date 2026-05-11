import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/auth': 'http://localhost:8000',
      '/conversations': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/query': 'http://localhost:8000',
      '/paper': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
