'use client';

import { useEffect, useState } from 'react';
import { api, HealthResponse, ModelInfoResponse, MetricsResponse, EvaluationResultsResponse } from '@/lib/api';
import Link from 'next/link';

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [evalRes, setEvalRes] = useState<EvaluationResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [h, mi, me, ev] = await Promise.all([
        api.getHealth().catch(() => null),
        api.getModelInfo().catch(() => null),
        api.getMetrics().catch(() => null),
        api.getEvaluationResults().catch(() => null),
      ]);
      setHealth(h);
      setModelInfo(mi);
      setMetrics(me);
      setEvalRes(ev);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div className="page-body"><div className="spinner" /> Loading dashboard...</div>;

  const isHealthy = health?.status === 'ok';

  return (
    <>
      <header className="page-header">
        <div className="page-header-inner">
          <div>
            <h1 className="page-title">SOC Dashboard</h1>
            <p className="page-subtitle">System Status and Analytics overview</p>
          </div>
          <div className={`badge ${isHealthy ? 'badge-ok' : 'badge-error'}`}>
            <div className="badge-dot" />
            {isHealthy ? 'System Online' : 'System Offline'}
          </div>
        </div>
      </header>

      <div className="page-body">
        <div className="grid-4 mb-5">
          <div className="metric-card">
            <div className="metric-label">LLM Status</div>
            <div className="metric-value">{modelInfo?.model_loaded ? 'Loaded' : 'Unloaded'}</div>
            <div className="metric-sub">{modelInfo?.model_id || 'Unknown model'}</div>
            <div className="metric-icon">🧠</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">RAG Index</div>
            <div className="metric-value">{modelInfo?.rag_chunks || 0}</div>
            <div className="metric-sub">Indexed Chunks</div>
            <div className="metric-icon">📚</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Total Requests</div>
            <div className="metric-value">{metrics?.total_requests || 0}</div>
            <div className="metric-sub">Since startup</div>
            <div className="metric-icon">🌐</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Error Rate</div>
            <div className="metric-value">{((metrics?.error_rate || 0) * 100).toFixed(1)}%</div>
            <div className="metric-sub">Mean latency: {metrics?.mean_latency_ms || 0}ms</div>
            <div className="metric-icon">⚡</div>
          </div>
        </div>

        <div className="grid-2">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Model Information</h2>
            </div>
            <table className="data-table">
              <tbody>
                <tr>
                  <td>Base Model</td>
                  <td className="text-cyan">{modelInfo?.model_id || '-'}</td>
                </tr>
                <tr>
                  <td>Parameters</td>
                  <td>{modelInfo?.parameter_count_m ? `${modelInfo.parameter_count_m}M` : '-'}</td>
                </tr>
                <tr>
                  <td>Device</td>
                  <td>{modelInfo?.device || '-'}</td>
                </tr>
                <tr>
                  <td>Precision</td>
                  <td>{modelInfo?.half_precision ? 'FP16' : 'FP32'}</td>
                </tr>
                <tr>
                  <td>Embedding Model</td>
                  <td>{modelInfo?.embedding_model_id || '-'}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Evaluation Summary</h2>
              <Link href="/evaluation" className="btn btn-secondary btn-sm">View Full Details</Link>
            </div>
            {evalRes?.available ? (
              <table className="data-table">
                <tbody>
                  <tr>
                    <td>Perplexity Delta</td>
                    <td className="text-green">{evalRes.ppl_delta?.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Adapted PPL</td>
                    <td>{evalRes.adapted_ppl?.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>MCQ Accuracy</td>
                    <td>{evalRes.adapted_mcq_accuracy ? (evalRes.adapted_mcq_accuracy * 100).toFixed(1) + '%' : '-'}</td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <div className="alert alert-info">
                Evaluation results not available. Run Phase 6 benchmark.
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
