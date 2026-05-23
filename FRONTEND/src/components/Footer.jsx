import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.content}>
        <div className={styles.brand}>
          <span className={styles.logo}>🛋️ FurniAI</span>
          <p className={styles.tagline}>AI-Powered Furniture Recommendations</p>
        </div>
        <div className={styles.info}>
          <p>Built by <strong>Tushar Singh Rajput</strong> — Team Lambda A</p>
          <p className={styles.tech}>React 19 • FastAPI • ChromaDB • Sentence Transformers</p>
        </div>
      </div>
      <div className={styles.border} />
      <p className={styles.copy}>© 2026 GenAI Recommender System — College Project</p>
    </footer>
  );
}
