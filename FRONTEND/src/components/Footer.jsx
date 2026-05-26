import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.content}>
        <div className={styles.brand}>
          <span className={styles.logo}>RoomSense</span>
          <p className={styles.tagline}>Hyper-personalized furniture recommendations</p>
        </div>
        <div className={styles.info}>
          <p>Built by <strong>Tushar Singh Rajput</strong> - Team Lambda A</p>
          <p className={styles.tech}>React 19 | FastAPI | ChromaDB | Sentence Transformers</p>
        </div>
      </div>
      <div className={styles.border} />
      <p className={styles.copy}>Copyright 2026 RoomSense Recommender System - College Project</p>
    </footer>
  );
}
