import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Aceternity UI components import from "@/lib/utils" and
    // "@/components/ui/..." — standard shadcn-style alias to src/.
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
})
