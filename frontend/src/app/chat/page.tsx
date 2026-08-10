'use client';

import { useState, useRef, useEffect } from 'react';
import { api, EvidenceSource } from '@/lib/api';
import EvidencePanel from '@/components/EvidencePanel';

interface ChatMessage {
  id: string;
  role: 'user' | 'ai';
  content: string;
  evidence?: EvidenceSource[];
  sufficient?: boolean;
  latency?: number;
  model?: string;
  confidence?: string;
}

export default function CyberChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await api.cyberChat(userMessage.content);
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'ai',
        content: res.answer,
        evidence: res.sources,
        sufficient: res.evidence_sufficient,
        latency: res.latency_ms,
        model: res.model,
        confidence: res.confidence,
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'ai',
          content: `Error: ${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-col" style={{ height: '100vh' }}>
      <header className="page-header" style={{ flexShrink: 0 }}>
        <div className="page-header-inner pb-4">
          <div>
            <h1 className="page-title">Cyber Chat</h1>
            <p className="page-subtitle">RAG-grounded defensive cybersecurity assistant</p>
          </div>
        </div>
      </header>

      <div className="chat-window">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-muted">
            Ask a cybersecurity question to begin.
          </div>
        )}
        
        {messages.map((m) => (
          <div key={m.id} className={`chat-message ${m.role}`}>
            <div className={`chat-avatar ${m.role}-avatar`}>
              {m.role === 'user' ? 'U' : 'AI'}
            </div>
            <div className="flex-col">
              <div className="chat-bubble whitespace-pre-wrap">
                {m.content}
              </div>
              
              {m.role === 'ai' && m.latency && (
                <div className="chat-bubble-meta">
                  <span className="badge badge-cyan">{m.model}</span>
                  <span>{m.latency}ms</span>
                  {m.confidence && (
                    <span className={`badge ${m.confidence === 'high' ? 'badge-ok' : m.confidence === 'medium' ? 'badge-warn' : 'badge-error'}`}>
                      {m.confidence} confidence
                    </span>
                  )}
                  {m.sufficient === false && (
                    <span className="text-amber ml-2">Insufficient evidence</span>
                  )}
                </div>
              )}
              
              {m.role === 'ai' && m.evidence && m.evidence.length > 0 && (
                <EvidencePanel evidence={m.evidence} />
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="chat-message ai">
            <div className="chat-avatar ai-avatar">AI</div>
            <div className="chat-bubble">
              <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <form onSubmit={handleSubmit} className="chat-input-row">
          <textarea
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about vulnerabilities, best practices, or defensive strategies..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            rows={2}
          />
          <button type="submit" className="btn btn-primary" disabled={!input.trim() || loading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
