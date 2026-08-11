'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const nav = [
  { href: '/',              label: 'Dashboard',      icon: '⊞' },
  { href: '/chat',          label: 'Cyber Chat',     icon: '💬' },
  { href: '/threat',        label: 'Threat Analysis',icon: '🛡' },
  { href: '/vulnerability', label: 'Vulnerability',  icon: '🔍' },
  { href: '/document',      label: 'Doc Analyzer',   icon: '📄' },
  { href: '/evaluation',    label: 'Evaluation',     icon: '📊' },
  { href: '/report',        label: 'Report',         icon: '📋' },
  { href: '/demo',          label: 'Live Comparison',icon: '⚡' },
];

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-mark">
          <div className="sidebar-logo-icon">🔐</div>
          <div>
            <div className="sidebar-logo-text">CyberAdapt</div>
            <div className="sidebar-logo-sub">LLM Platform</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="sidebar-section-label">Navigation</div>
        {nav.map(({ href, label, icon }) => (
          <Link
            key={href}
            href={href}
            className={`nav-item${pathname === href ? ' active' : ''}`}
          >
            <span className="nav-icon">{icon}</span>
            {label}
          </Link>
        ))}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="phase-badge">Phase 11</div>
        <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-3)' }}>
          Defensive use only
        </div>
      </div>
    </aside>
  );
}
