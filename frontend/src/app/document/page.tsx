'use client';

import { useState, useRef } from 'react';
import { api, DocumentAnalysisResponse } from '@/lib/api';
import EvidencePanel from '@/components/EvidencePanel';

export default function DocumentAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DocumentAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.uploadDocument(file);
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
            <h1 className="page-title">Document Analyzer</h1>
            <p className="page-subtitle">Extract security insights from TXT or PDF documents</p>
          </div>
        </div>
      </header>

      <div className="page-body">
        <div className="grid-2">
          {/* Input Panel */}
          <div className="card h-fit">
            <h2 className="card-title">Upload Security Document</h2>
            
            <div 
              className="drop-zone mb-4"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{ display: 'none' }} 
                accept=".txt,.pdf,.md,.csv,.json,.jsonl"
              />
              <div className="drop-zone-icon">📄</div>
              {file ? (
                <>
                  <div className="drop-zone-text font-bold text-cyan">{file.name}</div>
                  <div className="drop-zone-sub">{(file.size / 1024).toFixed(1)} KB</div>
                </>
              ) : (
                <>
                  <div className="drop-zone-text">Click to upload or drag & drop</div>
                  <div className="drop-zone-sub">TXT, PDF, MD (max 2MB)</div>
                </>
              )}
            </div>

            <button 
              className="btn btn-primary w-full justify-center" 
              onClick={handleAnalyze}
              disabled={loading || !file}
            >
              {loading ? <span className="spinner" /> : 'Analyze Document'}
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
                  <h2 className="card-title mb-0">Document Summary</h2>
                  <div className="flex gap-2 items-center">
                    <span className="badge badge-cyan">{result.model}</span>
                  </div>
                </div>
                
                <div className="mt-4">
                  <div className="text-1 mb-4 leading-relaxed">{result.summary}</div>
                  
                  <div className="grid-2 mb-0">
                    <div>
                      <div className="form-label">Characters Scanned</div>
                      <span className="font-mono">{result.char_count}</span>
                    </div>
                    <div>
                      <div className="form-label">Confidence</div>
                      <span className={`badge ${result.confidence === 'high' ? 'badge-ok' : result.confidence === 'medium' ? 'badge-warn' : 'badge-error'}`}>
                        {result.confidence}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {result.threats.length > 0 && (
                <div className="card">
                  <h2 className="card-title">Identified Threats</h2>
                  <div className="flex flex-wrap gap-2">
                    {result.threats.map((t, i) => (
                      <span key={i} className="badge badge-warn">{t}</span>
                    ))}
                  </div>
                </div>
              )}

              {result.vulnerabilities.length > 0 && (
                <div className="card">
                  <h2 className="card-title">Vulnerabilities Detected</h2>
                  <div className="flex flex-wrap gap-2">
                    {result.vulnerabilities.map((v, i) => (
                      <span key={i} className="badge badge-error">{v}</span>
                    ))}
                  </div>
                </div>
              )}

              {result.suspicious_indicators.length > 0 && (
                <div className="card">
                  <h2 className="card-title">Suspicious Indicators (IoCs)</h2>
                  <ul className="bullet-list">
                    {result.suspicious_indicators.map((ioc, i) => <li key={i} className="font-mono">{ioc}</li>)}
                  </ul>
                </div>
              )}

              <div className="card">
                <h2 className="card-title">Recommendations</h2>
                <ul className="bullet-list">
                  {result.recommendations.map((rec, i) => <li key={i}>{rec}</li>)}
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
