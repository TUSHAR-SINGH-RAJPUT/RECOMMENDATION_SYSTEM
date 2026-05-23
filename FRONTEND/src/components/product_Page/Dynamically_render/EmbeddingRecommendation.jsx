import { useEffect, useState } from "react";
import { getEmbeddingRecommendations } from "../../../api/recommenderApi";
import RecommendationCard from "../../recommendation/RecommendationCard";

const EmbeddingRecommendation = ({ query }) => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query) return;
    // setLoading(true);
    getEmbeddingRecommendations(query).then((res) => {
      setResults(res);
      setLoading(false);
    });
  }, [query]);

  if (loading) return <p>Loading embedding recommendations...</p>;

  return (
    <div>
      <h2>Embedding Recommendations</h2>
      <div style={{ display: "grid", gap: "15px" }}>
        {results.map((item, idx) => (
          <RecommendationCard key={idx} item={item} />
        ))}
      </div>
    </div>
  );
};

export default EmbeddingRecommendation;
