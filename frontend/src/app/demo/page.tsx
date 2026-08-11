'use client';

import { useState, useEffect } from 'react';
import { api, DemoQuestion, CompareResponse, EvidenceSource } from '@/lib/api';

// ── Small reusable components ────────────────────────────────────────────────

function AnswerCard({
  label,
  modelName,
  answer,
  latencyMs,
  error,
  badge,
  badgeClass,
  adapted,
}: {
  label: string;
  modelName: string;
  answer: string;
  latencyMs: number;
  error?: string | null;
  badge?: string;
  badgeClass?: string;
  adapted?: boolean;
}) {
  return (
    <div className={`card h-full flex-col gap-3 ${adapted ? 'border-cyan' : ''}`}>
      <div className="card-header mb-0">
        <div>
          <div className="form-label mb-1">{label}</div>
          <div className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {modelName}
          </div>
        </div>
        <div className="flex gap-2 items-center">
          {badge && <span className={`badge ${badgeClass ?? 'badge-info'}`}>{badge}</span>}
          <span className="badge badge-info">{latencyMs.toFixed(0)}ms</span>
        </div>
      </div>

      {error ? (
        <div className="alert alert-error mt-2">
          <strong>Model Error:</strong> {error}
        </div>
      ) : (
        <div
          className="mt-2 p-3 rounded text-2 leading-relaxed"
          style={{
            background: 'rgba(0,0,0,0.2)',
            border: '1px solid var(--border)',
            whiteSpace: 'pre-wrap',
            minHeight: 120,
          }}
        >
          {answer || <span className="text-muted italic">No response generated.</span>}
        </div>
      )}
    </div>
  );
}

function EvidenceSection({ sources, sufficient }: { sources: EvidenceSource[]; sufficient: boolean }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  return (
    <div className="card">
      <button
        className="flex w-full justify-between items-center"
        onClick={() => setOpen(o => !o)}
        style={{ background: 'none', border: 'none', cursor: 'pointer' }}
      >
        <h2 className="card-title mb-0">
          📚 Retrieved Evidence
          <span className={`badge ml-2 ${sufficient ? 'badge-ok' : 'badge-warn'}`}>
            {sufficient ? 'Sufficient' : 'Insufficient'}
          </span>
        </h2>
        <span className="text-muted">{open ? '▲' : '▼'} {sources.length} source{sources.length !== 1 ? 's' : ''}</span>
      </button>

      {open && (
        <div className="mt-3 flex-col gap-3">
          {sources.map((s, i) => (
            <div key={i} className="p-3 rounded" style={{ background: 'rgba(0,212,255,0.05)', border: '1px solid var(--border)' }}>
              <div className="flex justify-between items-center mb-1">
                <span className="badge badge-cyan">{s.document_type ?? 'doc'}</span>
                <span className="text-muted" style={{ fontSize: 11 }}>
                  Score: {s.score !== undefined ? s.score.toFixed(3) : 'N/A'}
                </span>
              </div>
              <div className="font-bold text-xs mb-1">{s.source}</div>
              {s.topic && <div className="text-muted text-xs mb-1">Topic: {s.topic}</div>}
              <div className="text-2 leading-relaxed" style={{ fontSize: 12 }}>{s.text_preview}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DemoPage() {
  const [demoQuestions, setDemoQuestions] = useState<DemoQuestion[]>([]);
  const [customQuestion, setCustomQuestion] = useState('');
  const [activeQuestion, setActiveQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDemoQuestions().then(setDemoQuestions).catch(console.error);
  }, []);

  const handleRun = async (question: string) => {
    if (!question.trim()) return;
    setActiveQuestion(question.trim());
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.compareModels(question.trim(), 3);
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <header className="page-header">
        <div className="page-header-inner">
          <div>
            <h1 className="page-title">Live Model Comparison</h1>
            <p className="page-subtitle">Base LLM vs CyberAdapt-LLM — honest side-by-side evaluation</p>
          </div>
          {result && (
            <div className="flex gap-2 items-center">
              <span className="badge badge-info">
                Total: {result.total_latency_ms.toFixed(0)}ms
              </span>
            </div>
          )}
        </div>
      </header>

      <div className="page-body">
        {/* Pre-designed questions */}
        <div className="card mb-5">
          <h2 className="card-title">📋 Demonstration Questions</h2>
          <p className="text-muted mb-4" style={{ fontSize: 13 }}>
            These questions are carefully designed to expose differences in cybersecurity domain knowledge.
            Results are reported honestly — if the adapted model underperforms, it will show.
          </p>
          <div className="flex-col gap-2">
            {demoQuestions.map(q => (
              <button
                key={q.id}
                className={`demo-question-btn ${activeQuestion === q.question ? 'active' : ''}`}
                onClick={() => handleRun(q.question)}
                disabled={loading}
              >
                <div className="flex justify-between items-start gap-3">
                  <div className="flex-col gap-1 text-left">
                    <span className="badge badge-cyan" style={{ fontSize: 10, marginBottom: 4 }}>
                      {q.category}
                    </span>
                    <div className="text-1" style={{ fontSize: 13 }}>{q.question}</div>
                    <div className="text-muted" style={{ fontSize: 11 }}>
                      💡 {q.rationale}
                    </div>
                  </div>
                  <span className="text-cyan" style={{ flexShrink: 0 }}>▶</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Custom question */}
        <div className="card mb-5">
          <h2 className="card-title">✏️ Custom Question</h2>
          <div className="flex gap-3 items-end">
            <div className="form-group flex-1 mb-0">
              <textarea
                className="form-textarea"
                placeholder="Enter your own cybersecurity question..."
                value={customQuestion}
                onChange={e => setCustomQuestion(e.target.value)}
                rows={3}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={() => handleRun(customQuestion)}
              disabled={loading || !customQuestion.trim()}
              style={{ flexShrink: 0 }}
            >
              {loading ? <span className="spinner" /> : 'Compare →'}
            </button>
          </div>
          {error && (
            <div className="alert alert-error mt-3">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        {/* Loading state */}
        {loading && (
          <div className="card text-center py-8">
            <div className="spinner mx-auto mb-3" style={{ width: 32, height: 32 }} />
            <div className="text-1">Querying both models simultaneously...</div>
            <div className="text-muted mt-1" style={{ fontSize: 12 }}>
              This may take 30–90 seconds depending on hardware.
            </div>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <>
            {/* Active question banner */}
            <div
              className="mb-4 p-3 rounded"
              style={{ background: 'rgba(0,212,255,0.08)', border: '1px solid var(--cyan)', borderRadius: 8 }}
            >
              <div className="text-muted text-xs mb-1">Comparing models on:</div>
              <div className="text-1 font-medium">{result.question}</div>
            </div>

            {/* Side-by-side answers */}
            <div className="grid-2 mb-4">
              <AnswerCard
                label="🤖 Base Model"
                modelName={result.base.model}
                answer={result.base.answer}
                latencyMs={result.base.latency_ms}
                error={result.base.error}
              />
              <AnswerCard
                label="🛡️ CyberAdapt-LLM"
                modelName={result.adapted.model}
                answer={result.adapted.answer}
                latencyMs={result.adapted.latency_ms}
                error={result.adapted.error}
                badge={result.adapted.adapted_model_available ? 'Adapted' : 'Base Fallback'}
                badgeClass={result.adapted.adapted_model_available ? 'badge-ok' : 'badge-warn'}
                adapted
              />
            </div>

            {/* Evidence */}
            <div className="mb-4">
              <EvidenceSection
                sources={result.evidence.sources}
                sufficient={result.evidence.evidence_sufficient}
              />
            </div>

            {/* Difference summary */}
            <div className="card mb-4">
              <h2 className="card-title">🔍 Why the Answers Differ</h2>
              {!result.adapted.adapted_model_available && (
                <div className="alert alert-warn mb-3">
                  <strong>Note:</strong> No adapted model found at <code>models/adapted/</code>.
                  Both answers above were generated by the same base model. Run Phase 5 training
                  to produce a genuinely adapted model for meaningful comparison.
                </div>
              )}
              <div
                className="p-3 rounded text-2 leading-relaxed"
                style={{
                  background: 'rgba(0,0,0,0.2)',
                  border: '1px solid var(--border)',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {result.difference_summary}
              </div>
            </div>

            {/* Evaluation stats */}
            <EvaluationStats />

            {/* Disclaimer */}
            <div className="mt-4 p-3 rounded text-muted" style={{ fontSize: 11, border: '1px dashed var(--border)' }}>
              ⚠️ {result.disclaimer}
            </div>
          </>
        )}
      </div>
    </>
  );
}

// ── Evaluation stats widget ───────────────────────────────────────────────────

function EvaluationStats() {
  const [data, setData] = useState<import('@/lib/api').EvaluationResultsResponse | null>(null);

  useEffect(() => {
    api.getEvaluationResults().then(setData).catch(() => setData(null));
  }, []);

  if (!data?.available) {
    return (
      <div className="card">
        <h2 className="card-title">📈 Evaluation Statistics</h2>
        <div className="text-muted text-sm">
          No evaluation results available. Run <code>evaluation/benchmark_runner.py</code> (Phase 6) to generate metrics.
        </div>
      </div>
    );
  }

  const fmt = (v?: number) => v !== undefined ? (v * 100).toFixed(1) + '%' : 'N/A';

  return (
    <div className="card">
      <h2 className="card-title">📈 Evaluation Statistics</h2>
      {data.caveat && (
        <div className="alert alert-warn mb-3"><strong>Note:</strong> {data.caveat}</div>
      )}
      <div className="grid-3">
        <div className="metric-card">
          <div className="metric-label">MCQ Accuracy</div>
          <div className="metric-value text-cyan">{fmt(data.adapted_mcq_accuracy)}</div>
          <div className="metric-sub">Base: {fmt(data.base_mcq_accuracy)}</div>
          <div className="metric-icon">🎯</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Keyword Recall</div>
          <div className="metric-value text-cyan">{fmt(data.adapted_gen_recall)}</div>
          <div className="metric-sub">Base: {fmt(data.base_gen_recall)}</div>
          <div className="metric-icon">📊</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Perplexity (↓)</div>
          <div className="metric-value text-green">{data.adapted_ppl?.toFixed(2) ?? 'N/A'}</div>
          <div className="metric-sub">Base: {data.base_ppl?.toFixed(2) ?? 'N/A'}</div>
          <div className="metric-icon">📉</div>
        </div>
      </div>
    </div>
  );
}
