import styles from './RecommendationCard.module.css';

function getCategoryColor(category) {
  const colors = {
    bed: '#ff5ea8',
    chair: '#26f3ff',
    sofa: '#3fffb0',
    table: '#a78bfa',
    desk: '#5aa7ff',
    lighting: '#ffd166',
  };
  return colors[category?.toLowerCase()] || '#88a2b4';
}

function getScoreColor(score) {
  if (score >= 0.8) return '#3fffb0';
  if (score >= 0.5) return '#26f3ff';
  return '#ff5ea8';
}

export default function RecommendationCard({ product, rank }) {
  const rawScore = Number(product.score ?? product.final_score ?? 0);
  const score = Math.max(0, Math.min(1, rawScore));
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
