import styles from './RecommendationCard.module.css';

function getCategoryColor(category) {
  const colors = {
    bed: '#c47b67',
    chair: '#d7a15f',
    sofa: '#8fa06d',
    table: '#b86f52',
    desk: '#b08a61',
    lighting: '#f4c987',
  };
  return colors[category?.toLowerCase()] || '#a08770';
}

function getScoreColor(score) {
  if (score >= 0.8) return '#8fa06d';
  if (score >= 0.5) return '#d7a15f';
  return '#c47b67';
}

export default function RecommendationCard({ product, rank }) {
  const score = Number(product.score ?? product.final_score ?? 0);
  const catColor = getCategoryColor(product.category);
  const scoreColor = getScoreColor(score);
  const scorePercent = Math.max(0, Math.min(100, Math.round(score * 100)));

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
            <span className={styles.scoreLabel}>Recommendation Score</span>
            <span className={styles.scoreValue} style={{ color: scoreColor }}>
              {score.toFixed(3)}
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
