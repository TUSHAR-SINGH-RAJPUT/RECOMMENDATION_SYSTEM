// import styles from "./ProductCard.module.css";
// export default function ProductCard({ product }) {
//   return (
//     <>
//       <div className={styles.productCard} >
//         <div className={styles.productImage}>
//           <img src={product.image} alt={product.name} />
//         </div>
//         <div className={styles.productDetails}>
//           <h2>{product.product_name}</h2>
//           <p>{product.description}</p>
//         </div>
//       </div>
//     </>
//   );
// }

// import React from "react";
import styles from "./ProductCard.module.css";

const ProductCard = ({ product }) => {
  return (
    <div className={styles.card}>
      <h3 className={styles.name}>{product.name}</h3>
      <p className={styles.category}>{product.category}</p>
      <p className={styles.description}>{product.description}</p>
      <p className={styles.score}>⭐ {product.score}</p>
    </div>
  );
};

export default ProductCard;
