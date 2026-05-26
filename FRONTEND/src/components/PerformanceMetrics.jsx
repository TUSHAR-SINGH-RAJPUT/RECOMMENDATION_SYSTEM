import styles from './PerformanceMetrics.module.css';

function getTimeColor(ms) {
  if (ms < 200) return '#8fa06d';
  if (ms < 500) return '#d7a15f';
  return '#c47b67';
}

const MODEL_LABELS = {
  embedding: { name: 'Embedding (Content-Based)', icon: 'E', color: '#d7a15f' },
  collaborative: { name: 'Collaborative Filtering', icon: 'C', color: '#8fa06d' },
  hybrid: { name: 'Hybrid (CF + Embedding)', icon: 'H', color: '#b86f52' },
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
          <span>MS</span>
        </div>
        <div className={styles.info}>
          <span className={styles.label}>Response Time</span>
          <span className={styles.value} style={{ color: timeColor }}>{metrics.responseTime}ms</span>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.icon} style={{ background: 'rgba(176, 138, 97, 0.14)' }}>
          <span>N</span>
        </div>
        <div className={styles.info}>
          <span className={styles.label}>Results Found</span>
          <span className={styles.value} style={{ color: '#b08a61' }}>{metrics.resultCount}</span>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.icon} style={{ background: 'rgba(148, 163, 184, 0.1)' }}>
          <span>T</span>
        </div>
        <div className={styles.info}>
          <span className={styles.label}>Queried At</span>
          <span className={styles.value} style={{ color: '#d1bda6' }}>{metrics.timestamp}</span>
        </div>
      </div>
    </div>
  );
}
