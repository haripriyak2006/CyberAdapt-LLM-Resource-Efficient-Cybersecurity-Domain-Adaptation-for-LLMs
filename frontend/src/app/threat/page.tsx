'use client';

import { useState } from 'react';
import { api, ThreatResponse } from '@/lib/api';
import EvidencePanel from '@/components/EvidencePanel';

export default function ThreatAnalysis() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ThreatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.analyzeThreat(input);
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <header className="page-header">
        <div className="page-header-inner">
          <div>
            <h1 className="page-title">Threat Analysis</h1>
            <p className="page-subtitle">Analyze security incidents and extract indicators</p>
          </div>
        </div>
      </header>

      <div className="page-body">
        <div className="grid-2">
          {/* Input Panel */}
          <div className="card h-fit">
            <h2 className="card-title">Incident Description</h2>
            <div className="form-group">
              <textarea
                className="form-textarea"
                placeholder="Paste security incident logs or description here..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                rows={8}
              />
            </div>
            <button 
              className="btn btn-primary w-full justify-center" 
              onClick={handleAnalyze}
              disabled={loading || !input.trim()}
            >
              {loading ? <span className="spinner" /> : 'Analyze Threat'}
            </button>
            
            {error && (
              <div className="alert alert-error mt-4">
                <strong>Error:</strong> {error}
              </div>
            )}
          </div>

          {/* Results Panel */}
          {result && (
            <div className="flex-col gap-4">
              <div className="card">
                <div className="card-header mb-0">
                  <h2 className="card-title mb-0">Analysis Results</h2>
                  <div className="flex gap-2 items-center">
                    <span className="badge badge-cyan">{result.model}</span>
                    <span className="text-muted" style={{ fontSize: 11 }}>{result.latency_ms}ms</span>
                  </div>
                </div>
                
                <div className="mt-4">
                  <div className="form-label">Threat Classification</div>
                  <div className="text-1 font-medium mb-4">{result.threat_type}</div>
                  
                  <div className="form-label">Attack Technique</div>
                  <div className="text-1 mb-4">{result.attack_technique}</div>
                  
                  <div className="form-label">Potential Impact</div>
                  <div className="text-2 mb-4">{result.potential_impact}</div>
                  
                  <div className="grid-2 mb-4">
                    <div>
                      <div className="form-label">Confidence</div>
                      <span className={`badge ${result.confidence === 'high' ? 'badge-ok' : result.confidence === 'medium' ? 'badge-warn' : 'badge-error'}`}>
                        {result.confidence}
                      </span>
                    </div>
                    <div>
                      <div className="form-label">Indicators Extracted</div>
                      <span className="badge badge-info">{result.indicators.length}</span>
                    </div>
                  </div>
                </div>
              </div>

              {result.indicators.length > 0 && (
                <div className="card">
                  <h2 className="card-title">Indicators of Compromise (IoCs)</h2>
                  <ul className="bullet-list">
                    {result.indicators.map((ioc, i) => <li key={i}>{ioc}</li>)}
                  </ul>
                </div>
              )}

              <div className="card">
                <h2 className="card-title">Defensive Actions</h2>
                <ul className="bullet-list">
                  {result.defensive_actions.map((action, i) => <li key={i}>{action}</li>)}
                </ul>
              </div>

              <EvidencePanel evidence={result.evidence} />
            </div>
          )}
        </div>
      </div>
    </>
  );
}
