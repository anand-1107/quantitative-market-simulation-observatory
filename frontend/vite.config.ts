import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true },
  build: {
    rollupOptions: {
      output: { manualChunks(id) {
        if (id.includes('/node_modules/recharts/')) return 'vendor-charts'
        if (id.includes('/node_modules/lucide-react/')) return 'vendor-icons'
      } },
    },
  },
})
