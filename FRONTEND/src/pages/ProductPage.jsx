import { useState } from "react";
import ProductCard from "../components/product/ProductCard";
import style from "./ProductPage.module.css";
import productsData from "../data/embedding_metadata.json";

export default function ProductPage() {
  const [showFilter, setShowFilter] = useState(false);

  // Store products in state
  const [products, setProducts] = useState(productsData);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);

  const productsPerPage = 20;

  function handleFilterClick() {
    setShowFilter(!showFilter);
  }

  // Sort Low to High
  function sortLowToHigh() {
    const sortedProducts = [...products].sort(
      (a, b) => a.price - b.price
    );

    setProducts(sortedProducts);
    setCurrentPage(1);
  }

  // Sort High to Low
  function sortHighToLow() {
    const sortedProducts = [...products].sort(
      (a, b) => b.price - a.price
    );

    setProducts(sortedProducts);
    setCurrentPage(1);
  }

  // Calculate indexes
  const startIndex = (currentPage - 1) * productsPerPage;
  const endIndex = startIndex + productsPerPage;

  // Current page products
  const currentProducts = products.slice(startIndex, endIndex);

  // Total pages
  const totalPages = Math.ceil(products.length / productsPerPage);

  return (
    <div className={style.productPageContainer}>
      <div className={style.filterContainer}>
        <button
          className={style.filterBtn}
          onClick={handleFilterClick}
        >
          Filter
        </button>

        <button className={`${style.sortBtn} ${style.filterBtn}`}>
          Sort
        </button>
      </div>

      {/* Filter Dropdown */}
      {showFilter && (
        <div className={style.filterDropDown}>
          <button onClick={sortLowToHigh}>
            Low to High
          </button>

          <button onClick={sortHighToLow}>
            High to Low
          </button>

          <button>Top Rated</button>
        </div>
      )}

      {/* Product Grid */}
      <div className={style.productGrid}>
        {currentProducts.map((product) => (
          <ProductCard
            key={product.product_id}
            product={product}
          />
        ))}
      </div>

      {/* Pagination */}
      <div className={style.pagination}>
        <button
          disabled={currentPage === 1}
          onClick={() =>
            setCurrentPage(currentPage - 1)
          }
        >
          Previous
        </button>

        <span>
          Page {currentPage} of {totalPages}
        </span>

        <button
          disabled={currentPage === totalPages}
          onClick={() =>
            setCurrentPage(currentPage + 1)
          }
        >
          Next
        </button>
      </div>
    </div>
  );
}