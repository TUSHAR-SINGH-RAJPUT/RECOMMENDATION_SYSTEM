import { useState, useCallback } from 'react';
import {
  getEmbeddingRecommendations,
  getCollaborativeRecommendations,
  getHybridRecommendations,
} from '../api/recommenderApi';

export default function useRecommendations() {
  const [selectedModel, setSelectedModel] = useState('embedding');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [userId] = useState(() => Math.floor(Math.random() * 500) + 1);

  const fetchRecommendations = useCallback(async (query) => {
    if (!query?.trim()) return;

    setLoading(true);
    setError(null);

    try {
      let response;

      if (selectedModel === 'embedding') {
        response = await getEmbeddingRecommendations({ query });
      } else if (selectedModel === 'collaborative') {
        response = await getCollaborativeRecommendations({ user_id: userId, query });
      } else if (selectedModel === 'hybrid') {
        response = await getHybridRecommendations({ user_id: userId, query });
      } else {
        throw new Error(`Unknown model: ${selectedModel}`);
      }

      const products = response.data.recommendations || [];
      setResults(products);
      setMetrics({
        model: selectedModel,
        responseTime: response.responseTime,
        resultCount: products.length,
        timestamp: new Date().toLocaleTimeString(),
      });
    } catch (err) {
      console.error('Recommendation fetch error:', err);
      setError(err.message || 'Failed to fetch recommendations. Is the backend running?');
      setResults([]);
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  }, [selectedModel, userId]);

  return {
    selectedModel,
    setSelectedModel,
    results,
    loading,
    error,
    metrics,
    userId,
    fetchRecommendations,
  };
}
