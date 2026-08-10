/**
 * src/lib/api.ts
 * CyberAdapt-LLM API client — proxied through Next.js rewrites to FastAPI backend.
 * All endpoints use /api/... which Next.js forwards to http://localhost:8000/...
 */

const BASE = '';   // empty = same origin, Next.js rewrites handle proxy

// ── Generic fetch helper ────────────────────────────────────────────────────
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { const j = await res.json(); detail = j.detail ?? j.error ?? detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// ── Type definitions ────────────────────────────────────────────────────────
export interface HealthResponse {
  status: string;
  app: string;
  version: string;
  phase: number;
  env: string;
}

export interface ModelInfoResponse {
  model_id: string;
  model_loaded: boolean;
  parameter_count_m?: number;
  device?: string;
  half_precision?: boolean;
  load_time_s?: number;
  embedding_model_id?: string;
  rag_chunks?: number;
  adapted_model_path?: string;
}

export interface MetricsResponse {
  uptime_seconds: number;
  total_requests: number;
  total_errors: number;
  error_rate: number;
  mean_latency_ms: number;
  requests_by_path: Record<string, number>;
  errors_by_path: Record<string, number>;
  model_loaded: boolean;
  rag_loaded: boolean;
  rag_chunks: number;
  timestamp: string;
}

export interface EvaluationResultsResponse {
  available: boolean;
  generated_at?: string;
  caveat?: string;
  base_model?: string;
  adapted_model?: string;
  base_mcq_accuracy?: number;
  adapted_mcq_accuracy?: number;
  base_gen_recall?: number;
  adapted_gen_recall?: number;
  base_ppl?: number;
  adapted_ppl?: number;
  ppl_delta?: number;
  message?: string;
}

export interface EvidenceSource {
  source: string;
  topic?: string;
  document_type?: string;
  license?: string;
  score?: number;
  text_preview?: string;
}

export interface ChatResponse {
  response: string;
  model: string;
  latency_ms: number;
}

export interface CyberChatResponse {
  answer: string;
  sources: EvidenceSource[];
  evidence_sufficient: boolean;
  confidence: string;
  latency_ms: number;
  model: string;
  disclaimer: string;
}

export interface ThreatResponse {
  threat_type: string;
  indicators: string[];
  potential_impact: string;
  attack_technique: string;
  defensive_actions: string[];
  confidence: string;
  evidence: EvidenceSource[];
  evidence_sufficient: boolean;
  latency_ms: number;
  model: string;
  disclaimer: string;
}

export interface VulnResponse {
  vulnerability_summary: string;
  affected_component: string;
  severity: string;
  attack_vector: string;
  potential_impact: string;
  mitigation: string;
  evidence: EvidenceSource[];
  evidence_sufficient: boolean;
  confidence: string;
  latency_ms: number;
  model: string;
  disclaimer: string;
}

export interface DocumentAnalysisResponse {
  summary: string;
  threats: string[];
  vulnerabilities: string[];
  suspicious_indicators: string[];
  recommendations: string[];
  evidence: EvidenceSource[];
  evidence_sufficient: boolean;
  confidence: string;
  char_count: number;
  latency_ms: number;
  model: string;
  disclaimer: string;
}

export interface SecurityReport {
  report_id: string;
  generated_at: string;
  analyst: string;
  organization: string;
  executive_summary: string;
  threat_description: string;
  affected_assets: string[];
  indicators: string[];
  risk_assessment: string;
  mitre_mapping: string[];
  recommendations: string[];
  evidence: EvidenceSource[];
  limitations: string;
  confidence: string;
  latency_ms: number;
  model: string;
  disclaimer: string;
}

// ── API functions ───────────────────────────────────────────────────────────
export const api = {
  getHealth:           () => apiFetch<HealthResponse>('/health'),
  getModelInfo:        () => apiFetch<ModelInfoResponse>('/api/model/info'),
  getMetrics:          () => apiFetch<MetricsResponse>('/api/metrics'),
  getEvaluationResults:() => apiFetch<EvaluationResultsResponse>('/api/evaluation/results'),

  chat: (message: string, max_tokens = 200) =>
    apiFetch<ChatResponse>('/api/chat', {
      method: 'POST', body: JSON.stringify({ message, max_tokens }),
    }),

  cyberChat: (message: string, top_k = 3, max_new_tokens = 200) =>
    apiFetch<CyberChatResponse>('/api/cyber/chat', {
      method: 'POST', body: JSON.stringify({ message, top_k, max_new_tokens }),
    }),

  analyzeThreat: (description: string, top_k = 3) =>
    apiFetch<ThreatResponse>('/api/threat/analyze', {
      method: 'POST', body: JSON.stringify({ description, top_k }),
    }),

  analyzeVulnerability: (description: string, top_k = 3) =>
    apiFetch<VulnResponse>('/api/vulnerability/analyze', {
      method: 'POST', body: JSON.stringify({ description, top_k }),
    }),

  analyzeDocument: (content: string, filename = 'document.txt', top_k = 3) =>
    apiFetch<DocumentAnalysisResponse>('/api/document/analyze', {
      method: 'POST', body: JSON.stringify({ content, filename, top_k }),
    }),

  uploadDocument: async (file: File, top_k = 3): Promise<DocumentAnalysisResponse> => {
    const form = new FormData();
    form.append('file', file);
    form.append('top_k', String(top_k));
    const res = await fetch('/api/document/upload', { method: 'POST', body: form });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const j = await res.json(); detail = j.detail ?? detail; } catch {}
      throw new Error(detail);
    }
    return res.json();
  },

  generateReport: (
    incident_description: string,
    affected_assets: string[],
    analyst_name: string,
    organization: string,
    top_k = 3,
  ) =>
    apiFetch<SecurityReport>('/api/report/generate', {
      method: 'POST',
      body: JSON.stringify({ incident_description, affected_assets, analyst_name, organization, top_k }),
    }),
};
