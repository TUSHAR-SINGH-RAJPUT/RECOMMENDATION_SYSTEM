import styles from './ProductCard.module.css';

function extractPrice(description) {
  const match = description?.match(/price:\s*(\d+)/i);
  return match ? `$${match[1]}` : null;
}

function getCategoryColor(category) {
  const colors = {
    bed: '#ff5ea8',
    chair: '#26f3ff',
    sofa: '#3fffb0',
    table: '#a78bfa',
    desk: '#5aa7ff',
  };
  return colors[category?.toLowerCase()] || '#88a2b4';
}

export default function ProductCard({ product, onSelect }) {
  const price = extractPrice(product.description);
  const catColor = getCategoryColor(product.category);

  return (
    <button
      className={styles.card}
      type="button"
      onClick={() => onSelect?.(product)}
      aria-label={`Find products similar to ${product.product_name}`}
    >
      <div className={styles.cardInner}>
        <div className={styles.header}>
          <span className={styles.badge} style={{ background: `${catColor}20`, color: catColor, borderColor: `${catColor}40` }}>
            {product.category}
          </span>
          {price && <span className={styles.price}>{price}</span>}
        </div>

        <h3 className={styles.name}>{product.product_name}</h3>

        <p className={styles.description}>{product.description}</p>

        <div className={styles.footer}>
          <span className={styles.id}>{product.product_id}</span>
        </div>
      </div>
    </button>
  );
}
