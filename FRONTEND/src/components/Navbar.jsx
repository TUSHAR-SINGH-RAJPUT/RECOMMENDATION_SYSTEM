import { useState } from 'react';
import styles from './Navbar.module.css';

const MODEL_OPTIONS = [
  { value: 'embedding', label: 'Embedding', desc: 'Content-based semantic search', icon: 'E' },
  { value: 'hybrid', label: 'Hybrid', desc: 'CF + semantic + ranking', icon: 'H' },
  { value: 'collaborative', label: 'Collaborative', desc: 'User behavior patterns', icon: 'C' },
];

export default function Navbar({ selectedModel, onModelChange }) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const current = MODEL_OPTIONS.find((model) => model.value === selectedModel);

  return (
    <nav className={styles.navbar}>
      <div className={styles.logo}>
        <span className={styles.logoIcon}>AI</span>
        <span className={styles.logoText}>RoomSense</span>
      </div>

      <div className={styles.modelSelector}>
        <button
          className={styles.selectorBtn}
          onClick={() => setDropdownOpen(!dropdownOpen)}
          id="model-selector-btn"
          type="button"
        >
          <span className={styles.selectorIcon}>{current?.icon}</span>
          <span className={styles.selectorLabel}>{current?.label}</span>
          <span className={`${styles.chevron} ${dropdownOpen ? styles.chevronOpen : ''}`}>v</span>
        </button>

        {dropdownOpen && (
          <div className={styles.dropdown}>
            {MODEL_OPTIONS.map((option) => (
              <button
                key={option.value}
                className={`${styles.dropdownItem} ${selectedModel === option.value ? styles.active : ''}`}
                onClick={() => {
                  onModelChange(option.value);
                  setDropdownOpen(false);
                }}
                id={`model-option-${option.value}`}
                type="button"
              >
                <span className={styles.optIcon}>{option.icon}</span>
                <div className={styles.optInfo}>
                  <span className={styles.optLabel}>{option.label}</span>
                  <span className={styles.optDesc}>{option.desc}</span>
                </div>
                {selectedModel === option.value && <span className={styles.checkmark}>OK</span>}
              </button>
            ))}
          </div>
        )}
      </div>

      {dropdownOpen && <div className={styles.overlay} onClick={() => setDropdownOpen(false)} />}
    </nav>
  );
}
