import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/create-assets/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/patient': 'http://127.0.0.1:8000',
      '/save_record': 'http://127.0.0.1:8000',
      '/transcribe': 'http://127.0.0.1:8000',
      '/label_conversation': 'http://127.0.0.1:8000',
    },
  },
})
