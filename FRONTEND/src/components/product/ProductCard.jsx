export function ProductCard({ product }) {
  return (
    <>
      <div className="product-card">
        <div className="product-image">
          <img src={product.image} alt={product.name} />
        </div>
      </div>

    </>
  );
}
