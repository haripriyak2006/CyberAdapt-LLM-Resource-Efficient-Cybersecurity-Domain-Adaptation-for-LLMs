'use client';

import { useState } from 'react';
import { api, SecurityReport as ISecurityReport } from '@/lib/api';
import EvidencePanel from '@/components/EvidencePanel';

export default function SecurityReport() {
  const [incidentDesc, setIncidentDesc] = useState('');
  const [assets, setAssets] = useState('');
  const [analyst, setAnalyst] = useState('AI SOC Analyst');
  const [org, setOrg] = useState('CyberAdapt Internal');
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ISecurityReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!incidentDesc.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const assetList = assets.split(',').map(a => a.trim()).filter(Boolean);
      const res = await api.generateReport(incidentDesc, assetList, analyst, org);
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <>
      <header className="page-header" style={{ '@media print': { display: 'none' } } as any}>
        <div className="page-header-inner">
          <div>
            <h1 className="page-title">Security Report Generator</h1>
            <p className="page-subtitle">Generate structured executive and technical reports</p>
          </div>
        </div>
      </header>

      <div className="page-body">
        {/* Input Form (hidden when printing) */}
        <div className="card mb-5" style={{ '@media print': { display: 'none' } } as any}>
          <div className="grid-2">
            <div>
              <div className="form-group">
                <label className="form-label">Incident Description</label>
                <textarea
                  className="form-textarea"
                  value={incidentDesc}
                  onChange={(e) => setIncidentDesc(e.target.value)}
                  placeholder="Describe the incident..."
                  rows={5}
                />
              </div>
            </div>
            <div className="flex-col gap-3">
              <div className="form-group">
                <label className="form-label">Affected Assets (comma separated)</label>
                <input
                  type="text"
                  className="form-input"
                  value={assets}
                  onChange={(e) => setAssets(e.target.value)}
                  placeholder="e.g. SRV-01, DB-Primary, 10.0.0.4"
                />
              </div>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Analyst Name</label>
                  <input
                    type="text"
                    className="form-input"
                    value={analyst}
                    onChange={(e) => setAnalyst(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Organization</label>
                  <input
                    type="text"
                    className="form-input"
                    value={org}
                    onChange={(e) => setOrg(e.target.value)}
                  />
                </div>
              </div>
              <button 
                className="btn btn-primary mt-2 justify-center" 
                onClick={handleGenerate}
                disabled={loading || !incidentDesc.trim()}
              >
                {loading ? <span className="spinner" /> : 'Generate Formal Report'}
              </button>
            </div>
          </div>
          {error && (
            <div className="alert alert-error mt-4">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        {/* Generated Report View */}
        {result && (
          <div className="card bg-card print:bg-white print:text-black">
            <div className="flex justify-between items-start mb-5 border-b border-gray-700 pb-4">
              <div>
                <h1 className="text-2xl font-bold text-cyan uppercase tracking-wide">Incident Security Report</h1>
                <div className="text-sm text-gray-400 mt-1">ID: {result.report_id}</div>
              </div>
              <div className="text-right text-sm text-gray-400">
                <div><strong>Date:</strong> {new Date(result.generated_at).toLocaleString()}</div>
                <div><strong>Analyst:</strong> {result.analyst}</div>
                <div><strong>Org:</strong> {result.organization}</div>
              </div>
            </div>

            <div className="report-section">
              <div className="report-section-title">Executive Summary</div>
              <div className="report-section-body">{result.executive_summary}</div>
            </div>

            <div className="report-section">
              <div className="report-section-title">Threat Description</div>
              <div className="report-section-body">{result.threat_description}</div>
            </div>

            <div className="grid-2 section-gap">
              <div className="report-section">
                <div className="report-section-title">Affected Assets</div>
                <ul className="bullet-list">
                  {result.affected_assets.map((a, i) => <li key={i}>{a}</li>)}
                  {result.affected_assets.length === 0 && <li>None specified</li>}
                </ul>
              </div>

              <div className="report-section">
                <div className="report-section-title">Indicators of Compromise</div>
                <ul className="bullet-list">
                  {result.indicators.map((a, i) => <li key={i} className="font-mono">{a}</li>)}
                  {result.indicators.length === 0 && <li>None extracted</li>}
                </ul>
              </div>
            </div>

            <div className="grid-2 section-gap">
              <div className="report-section">
                <div className="report-section-title">Risk Assessment</div>
                <div className="report-section-body">{result.risk_assessment}</div>
              </div>
              
              <div className="report-section">
                <div className="report-section-title">MITRE ATT&CK Mapping</div>
                <ul className="bullet-list">
                  {result.mitre_mapping.map((m, i) => <li key={i}>{m}</li>)}
                  {result.mitre_mapping.length === 0 && <li>None mapped</li>}
                </ul>
              </div>
            </div>

            <div className="report-section">
              <div className="report-section-title">Recommended Defensive Actions</div>
              <ul className="bullet-list">
                {result.recommendations.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>

            <div className="mt-8 pt-4 border-t border-gray-700 text-xs text-gray-500">
              <p className="mb-2"><strong>Limitations:</strong> {result.limitations}</p>
              <p><strong>Disclaimer:</strong> {result.disclaimer}</p>
              <p className="mt-2">Generated by {result.model} (Confidence: {result.confidence})</p>
            </div>

            <div className="mt-5 text-right" style={{ '@media print': { display: 'none' } } as any}>
              <button className="btn btn-secondary" onClick={handlePrint}>Print / Save PDF</button>
            </div>
            
            <div className="mt-5" style={{ '@media print': { display: 'none' } } as any}>
              <EvidencePanel evidence={result.evidence} />
            </div>
          </div>
        )}
      </div>
    </>
  );
}
