import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, Vite serves the UI on :5173 and forwards /api to the Python server.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
