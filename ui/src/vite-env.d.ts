/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where the LangGraph server is; defaults to http://localhost:2024 (see src/config.ts). */
  readonly VITE_API_URL?: string
  /** The optional public CARTO basemap key; read only in src/map/basemap.ts. Unset means keyless tiles. */
  readonly VITE_CARTO_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
