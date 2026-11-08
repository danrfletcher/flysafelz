// The one browser setting: where the LangGraph server is. Nothing in this PR calls it; the chat PR will.
export const DEFAULT_API_URL = 'http://localhost:2024'

/** VITE_API_URL when it is set and not blank (a trailing slash dropped), else http://localhost:2024. Read per call so a test can stub it. */
export function apiUrl(): string {
  const set = import.meta.env.VITE_API_URL?.trim()
  return (set ? set : DEFAULT_API_URL).replace(/\/+$/, '')
}
