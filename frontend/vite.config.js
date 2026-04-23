import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/gnn-churn-portfolio/',  // Change to your GitHub repo name
})
