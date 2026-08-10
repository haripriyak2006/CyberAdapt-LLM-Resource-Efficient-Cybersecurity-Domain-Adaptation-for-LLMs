'use client';

import { useEffect, useState } from 'react';
import { api, EvaluationResultsResponse } from '@/lib/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

export default function Evaluation() {
  const [data, setData] = useState<EvaluationResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getEvaluationResults()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page-body"><div className="spinner" /> Loading evaluation results...</div>;

  if (!data?.available) {
    return (
      <div className="page-body">
        <div className="alert alert-info">
          Evaluation results not available. Please run the Phase 6 benchmark script.
        </div>
      </div>
    );
  }

  const chartData = [
    {
      name: 'MCQ Accuracy (%)',
      Base: (data.base_mcq_accuracy || 0) * 100,
      Adapted: (data.adapted_mcq_accuracy || 0) * 100,
    },
    {
      name: 'Keyword Recall (%)',
      Base: (data.base_gen_recall || 0) * 100,
      Adapted: (data.adapted_gen_recall || 0) * 100,
    }
  ];

  const radarData = [
    { subject: 'Accuracy', A: (data.adapted_mcq_accuracy || 0) * 100, B: (data.base_mcq_accuracy || 0) * 100, fullMark: 100 },
    { subject: 'Recall', A: (data.adapted_gen_recall || 0) * 100, B: (data.base_gen_recall || 0) * 100, fullMark: 100 },
    { subject: 'PPL Reduction', A: 100, B: (data.adapted_ppl && data.base_ppl) ? (data.adapted_ppl / data.base_ppl) * 100 : 0, fullMark: 100 },
  ];

  return (
    <>
      <header className="page-header">
        <div className="page-header-inner">
          <div>
            <h1 className="page-title">Model Evaluation</h1>
            <p className="page-subtitle">Base Model vs CyberAdapt-LLM Performance Metrics</p>
          </div>
        </div>
      </header>

      <div className="page-body">
        {data.caveat && (
          <div className="alert alert-warn mb-5">
            <strong>Note:</strong> {data.caveat}
          </div>
        )}

        <div className="grid-2 mb-5">
          <div className="card">
            <h2 className="card-title">Performance Comparison</h2>
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3c6e" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ backgroundColor: '#0b1628', borderColor: '#1e3c6e' }} />
                  <Legend />
                  <Bar dataKey="Base" fill="#5a7499" name={data.base_model || 'Base Model'} />
                  <Bar dataKey="Adapted" fill="#00d4ff" name={data.adapted_model || 'CyberAdapt'} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <h2 className="card-title">Capability Radar</h2>
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                  <PolarGrid stroke="#1e3c6e" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#5a7499' }} />
                  <Radar name={data.adapted_model || 'CyberAdapt'} dataKey="A" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.5} />
                  <Radar name={data.base_model || 'Base Model'} dataKey="B" stroke="#5a7499" fill="#5a7499" fillOpacity={0.5} />
                  <Legend />
                  <Tooltip contentStyle={{ backgroundColor: '#0b1628', borderColor: '#1e3c6e' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="grid-3">
          <div className="metric-card">
            <div className="metric-label">Perplexity (Lower is Better)</div>
            <div className="metric-value text-cyan">{data.adapted_ppl?.toFixed(2)}</div>
            <div className="metric-sub">Base: {data.base_ppl?.toFixed(2)}</div>
            <div className="metric-icon">📉</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">PPL Improvement</div>
            <div className="metric-value text-green">{data.ppl_delta?.toFixed(2)}</div>
            <div className="metric-sub">Absolute reduction</div>
            <div className="metric-icon">✨</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">MCQ Accuracy</div>
            <div className="metric-value text-cyan">{data.adapted_mcq_accuracy ? (data.adapted_mcq_accuracy * 100).toFixed(1) : 0}%</div>
            <div className="metric-sub">Base: {data.base_mcq_accuracy ? (data.base_mcq_accuracy * 100).toFixed(1) : 0}%</div>
            <div className="metric-icon">🎯</div>
          </div>
        </div>
      </div>
    </>
  );
}
