import { useState, useCallback } from 'react';
import {
  getEmbeddingRecommendations,
  getCollaborativeRecommendations,
  getHybridRecommendations,
  getConversationalRecommendations,
} from '../api/recommenderApi';

export default function useRecommendations() {
  const [selectedModel, setSelectedModel] = useState('embedding');
  const [results, setResults] = useState([]);
  const [aiResponse, setAiResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [userId] = useState(() => Math.floor(Math.random() * 500) + 1);

  const fetchRecommendations = useCallback(async (query) => {
    if (!query?.trim()) return;

    setLoading(true);
    setError(null);
    setAiResponse(null);

    try {
      let response;

      switch (selectedModel) {
        case 'embedding':
          response = await getEmbeddingRecommendations({ query });
          setResults(response.data.recommendations || []);
          break;

        case 'collaborative':
          response = await getCollaborativeRecommendations({
            user_id: userId,
            query,
          });
          setResults(response.data.recommendations || []);
          break;

        case 'hybrid':
          response = await getHybridRecommendations({
            user_id: userId,
            query,
          });
          setResults(response.data.recommendations || []);
          break;

        case 'conversational':
          response = await getConversationalRecommendations({
            user_id: userId,
            query,
          });
          setResults(response.data.products || []);
          setAiResponse(response.data.response || null);
          break;

        default:
          throw new Error(`Unknown model: ${selectedModel}`);
      }

      const products =
        selectedModel === 'conversational'
          ? response.data.products || []
          : response.data.recommendations || [];

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
    aiResponse,
    loading,
    error,
    metrics,
    userId,
    fetchRecommendations,
  };
}
