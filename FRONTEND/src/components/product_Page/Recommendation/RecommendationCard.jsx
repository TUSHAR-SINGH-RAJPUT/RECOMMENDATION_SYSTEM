// import React from "react";
import styles from "./RecommendationCard.module.css";

const RecommendationCard = ({ item }) => {
  return (
    <div className={styles.card}>
      <h3>{item.name}</h3>
      <p>{item.category}</p>
      <p>{item.description}</p>
      <p className={styles.score}>Score: {item.score}</p>
    </div>
  );
};

export default RecommendationCard;
