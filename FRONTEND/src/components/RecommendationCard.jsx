import styles from './RecommendationCard.module.css';

function getCategoryColor(category) {
  const colors = {
    bed: '#f472b6',
    chair: '#60a5fa',
    sofa: '#34d399',
    table: '#fbbf24',
    desk: '#a78bfa',
    lighting: '#fb923c',
  };
  return colors[category?.toLowerCase()] || '#94a3b8';
}

function getScoreColor(score) {
  if (score >= 0.8) return '#10b981';
  if (score >= 0.5) return '#fbbf24';
  return '#f87171';
}

export default function RecommendationCard({ product, rank }) {
  const catColor = getCategoryColor(product.category);
  const scoreColor = getScoreColor(product.score);
  const scorePercent = Math.round((product.score || 0) * 100);

  return (
    <div className={styles.card} style={{ animationDelay: `${rank * 0.08}s` }}>
      <div className={styles.rankBadge}>#{rank}</div>

      <div className={styles.cardInner}>
        <div className={styles.header}>
          <span className={styles.badge} style={{ background: `${catColor}20`, color: catColor, borderColor: `${catColor}40` }}>
            {product.category}
          </span>
          <span className={styles.pid}>{product.product_id}</span>
        </div>

        <h3 className={styles.name}>{product.product_name}</h3>

        {product.description && (
          <p className={styles.description}>{product.description}</p>
        )}

        <div className={styles.scoreSection}>
          <div className={styles.scoreHeader}>
            <span className={styles.scoreLabel}>Similarity Score</span>
            <span className={styles.scoreValue} style={{ color: scoreColor }}>
              {product.score?.toFixed(3)}
            </span>
          </div>
          <div className={styles.scoreBar}>
            <div
              className={styles.scoreFill}
              style={{ width: `${scorePercent}%`, background: scoreColor }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
