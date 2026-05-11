import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/auth': 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space',
      '/conversations': 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space',
      '/documents': 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space',
      '/query': 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space',
      '/paper': 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space',
      '/health': 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space',
    },
  },
})
