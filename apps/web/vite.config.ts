/// <reference types="vitest" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// VITE_BASE_PATH="/Human-possibility-engine/" for GitHub Pages; "/" (default) elsewhere.
// loadEnv also picks up VITE_-prefixed process.env variables, so no Node typings are needed.
export default defineConfig(({ mode }) => ({
  base: loadEnv(mode, ".", "VITE_").VITE_BASE_PATH || "/",
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, strictPort: true },
  preview: { host: "127.0.0.1", port: 4173 },
  build: {
    target: "es2022",
    sourcemap: true,
    chunkSizeWarningLimit: 300,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    restoreMocks: true,
    unstubEnvs: true,
  },
}));
