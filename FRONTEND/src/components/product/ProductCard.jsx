import styles from "./ProductCard.module.css";
export default function ProductCard({ product }) {
  return (
    <>
      <div className={styles.productCard} >
        <div className={styles.productImage}>
          <img src={product.image} alt={product.name} />
        </div>
        <div className={styles.productDetails}>
          <h2>{product.product_name}</h2>
          <p>{product.description}</p>
        </div>
      </div>
    </>
  );
}