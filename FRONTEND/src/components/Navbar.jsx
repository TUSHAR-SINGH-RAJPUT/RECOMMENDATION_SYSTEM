import { useState } from 'react';
import styles from './Navbar.module.css';

const MODEL_OPTIONS = [
  { value: 'embedding', label: 'Embedding', desc: 'Content-based semantic search', icon: '🧬' },
  { value: 'collaborative', label: 'Collaborative', desc: 'User behavior patterns', icon: '👥' },
  { value: 'hybrid', label: 'Hybrid', desc: 'Combined CF + Embedding', icon: '⚡' },
  { value: 'conversational', label: 'Conversational', desc: 'AI chat recommendations', icon: '💬' },
];

export default function Navbar({ selectedModel, onModelChange }) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const current = MODEL_OPTIONS.find(m => m.value === selectedModel);

  return (
    <nav className={styles.navbar}>
      <div className={styles.logo}>
        <span className={styles.logoIcon}>🛋️</span>
        <span className={styles.logoText}>FurniAI</span>
      </div>

      <div className={styles.modelSelector}>
        <button
          className={styles.selectorBtn}
          onClick={() => setDropdownOpen(!dropdownOpen)}
          id="model-selector-btn"
        >
          <span className={styles.selectorIcon}>{current?.icon}</span>
          <span className={styles.selectorLabel}>{current?.label}</span>
          <span className={`${styles.chevron} ${dropdownOpen ? styles.chevronOpen : ''}`}>▾</span>
        </button>

        {dropdownOpen && (
          <div className={styles.dropdown}>
            {MODEL_OPTIONS.map(opt => (
              <button
                key={opt.value}
                className={`${styles.dropdownItem} ${selectedModel === opt.value ? styles.active : ''}`}
                onClick={() => { onModelChange(opt.value); setDropdownOpen(false); }}
                id={`model-option-${opt.value}`}
              >
                <span className={styles.optIcon}>{opt.icon}</span>
                <div className={styles.optInfo}>
                  <span className={styles.optLabel}>{opt.label}</span>
                  <span className={styles.optDesc}>{opt.desc}</span>
                </div>
                {selectedModel === opt.value && <span className={styles.checkmark}>✓</span>}
              </button>
            ))}
          </div>
        )}
      </div>

      {dropdownOpen && <div className={styles.overlay} onClick={() => setDropdownOpen(false)} />}
    </nav>
  );
}
