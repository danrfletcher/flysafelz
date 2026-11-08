/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where the LangGraph server is; defaults to http://localhost:2024 (see src/config.ts). */
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
