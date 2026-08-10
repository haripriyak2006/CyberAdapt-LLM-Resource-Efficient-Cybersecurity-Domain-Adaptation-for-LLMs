import { EvidenceSource } from '@/lib/api';

interface EvidencePanelProps {
  evidence: EvidenceSource[];
}

export default function EvidencePanel({ evidence }: EvidencePanelProps) {
  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="evidence-panel mt-4">
      <div className="evidence-header">
        <span className="evidence-header-title">Retrieved Evidence</span>
        <span className="badge badge-info">{evidence.length} sources</span>
      </div>
      <div>
        {evidence.map((src, idx) => (
          <div key={idx} className="evidence-item">
            <div className="evidence-meta">
              <span className="evidence-source">{src.source}</span>
              {src.score !== undefined && (
                <span className="evidence-score">score: {src.score.toFixed(3)}</span>
              )}
              {src.topic && <span className="evidence-topic">{src.topic}</span>}
              {src.document_type && <span className="badge badge-info">{src.document_type}</span>}
            </div>
            <div className="evidence-text">
              {src.text_preview}...
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
