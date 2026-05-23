import styles from './PerformanceMetrics.module.css';

function getTimeColor(ms) {
  if (ms < 200) return '#10b981';
  if (ms < 500) return '#fbbf24';
  return '#f87171';
}

const MODEL_LABELS = {
  embedding: { name: 'Embedding (Content-Based)', icon: '🧬', color: '#00d4ff' },
  collaborative: { name: 'Collaborative Filtering', icon: '👥', color: '#a78bfa' },
  hybrid: { name: 'Hybrid (CF + Embedding)', icon: '⚡', color: '#fbbf24' },
  conversational: { name: 'Conversational RAG', icon: '💬', color: '#34d399' },
};

export default function PerformanceMetrics({ metrics }) {
  if (!metrics) return null;

  const model = MODEL_LABELS[metrics.model] || {};
  const timeColor = getTimeColor(metrics.responseTime);

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <div className={styles.icon} style={{ background: `${model.color}15` }}>
          <span>{model.icon}</span>
        </div>
        <div className={styles.info}>
          <span className={styles.label}>Active Model</span>
          <span className={styles.value} style={{ color: model.color }}>{model.name}</span>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.icon} style={{ background: `${timeColor}15` }}>
          <span>⚡</span>
        </div>
        <div className={styles.info}>
          <span className={styles.label}>Response Time</span>
          <span className={styles.value} style={{ color: timeColor }}>{metrics.responseTime}ms</span>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.icon} style={{ background: 'rgba(96, 165, 250, 0.1)' }}>
          <span>📦</span>
        </div>
        <div className={styles.info}>
          <span className={styles.label}>Results Found</span>
          <span className={styles.value} style={{ color: '#60a5fa' }}>{metrics.resultCount}</span>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.icon} style={{ background: 'rgba(148, 163, 184, 0.1)' }}>
          <span>🕐</span>
        </div>
        <div className={styles.info}>
          <span className={styles.label}>Queried At</span>
          <span className={styles.value} style={{ color: '#94a3b8' }}>{metrics.timestamp}</span>
        </div>
      </div>
    </div>
  );
}
