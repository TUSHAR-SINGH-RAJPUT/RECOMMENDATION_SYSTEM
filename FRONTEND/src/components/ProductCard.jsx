import styles from './ProductCard.module.css';

function extractPrice(description) {
  const match = description?.match(/price:\s*(\d+)/i);
  return match ? `$${match[1]}` : null;
}

function getCategoryColor(category) {
  const colors = {
    bed: '#c47b67',
    chair: '#d7a15f',
    sofa: '#8fa06d',
    table: '#b86f52',
    desk: '#b08a61',
  };
  return colors[category?.toLowerCase()] || '#a08770';
}

export default function ProductCard({ product }) {
  const price = extractPrice(product.description);
  const catColor = getCategoryColor(product.category);

  return (
    <div className={styles.card}>
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
    </div>
  );
}
