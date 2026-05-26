import { useState } from 'react';
import styles from './SearchBar.module.css';

const SUGGESTIONS = [
  'modern wooden sofa',
  'minimalist office chair',
  'luxury king size bed',
  'gaming desk with storage',
  'glass coffee table',
  'metal bookshelf',
];

export default function SearchBar({ onSearch, loading }) {
  const [value, setValue] = useState('');

  function handleSubmit(event) {
    event.preventDefault();
    if (value.trim()) onSearch(value.trim());
  }

  function handleSuggestion(text) {
    setValue(text);
    onSearch(text);
  }

  return (
    <div className={styles.wrapper}>
      <form className={styles.searchForm} onSubmit={handleSubmit}>
        <div className={styles.inputWrapper}>
          <span className={styles.searchIcon}>S</span>
          <input
            type="text"
            className={styles.input}
            placeholder="Describe the furniture you are looking for..."
            value={value}
            onChange={(event) => setValue(event.target.value)}
            id="search-input"
          />
        </div>
        <button
          type="submit"
          className={styles.searchBtn}
          disabled={loading || !value.trim()}
          id="search-button"
        >
          {loading ? <span className={styles.spinner} /> : 'Search'}
        </button>
      </form>

      <div className={styles.suggestions}>
        <span className={styles.sugLabel}>Try:</span>
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            className={styles.chip}
            onClick={() => handleSuggestion(suggestion)}
            type="button"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
