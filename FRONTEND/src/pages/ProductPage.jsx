// import { useState } from "react";
// import ProductCard from "../components/product_Page/ProductCard";
// import style from "./ProductPage.module.css";
// import productsData from "../data/embedding_metadata.json";
// import RecommendationCard from "../components/product_Page/Dynamically_render/Dynamically_render/CollaborativeRecommendation";
// import 
// export default function ProductPage() {
//   const [showFilter, setShowFilter] = useState(false);

//   // Store products in state
//   const [products, setProducts] = useState(productsData);

//   // Pagination state
//   const [currentPage, setCurrentPage] = useState(1);

//   const productsPerPage = 20;

//   function handleFilterClick() {
//     setShowFilter(!showFilter);
//   }

//   // Sort Low to High
//   function sortLowToHigh() {
//     const sortedProducts = [...products].sort(
//       (a, b) => a.price - b.price
//     );

//     setProducts(sortedProducts);
//     setCurrentPage(1);
//   }

//   // Sort High to Low
//   function sortHighToLow() {
//     const sortedProducts = [...products].sort(
//       (a, b) => b.price - a.price
//     );

//     setProducts(sortedProducts);
//     setCurrentPage(1);
//   }

//   // Calculate indexes
//   const startIndex = (currentPage - 1) * productsPerPage;
//   const endIndex = startIndex + productsPerPage;

//   // Current page products
//   const currentProducts = products.slice(startIndex, endIndex);

//   // Total pages
//   const totalPages = Math.ceil(products.length / productsPerPage);

//   return (
//     <div className={style.productPageContainer}>
//       <div className={style.filterContainer}>
//         <button
//           className={style.filterBtn}
//           onClick={handleFilterClick}
//         >
//           Filter
//         </button>

//         <button className={`${style.sortBtn} ${style.filterBtn}`}>
//           Sort
//         </button>
//       </div>

//       {/* Filter Dropdown */}
//       {showFilter && (
//         <div className={style.filterDropDown}>
//           <button onClick={sortLowToHigh}>
//             Low to High
//           </button>

//           <button onClick={sortHighToLow}>
//             High to Low
//           </button>

//           <button>Top Rated</button>
//         </div>
//       )}

//       {/* Product Grid */}
//       <div className={style.productGrid}>
//         {currentProducts.map((product) => (
//           <ProductCard
//             key={product.product_id}
//             product={product}
//           />
//         ))}
//       </div>

//       {/* Pagination */}
//       <div className={style.pagination}>
//         <button
//           disabled={currentPage === 1}
//           onClick={() =>
//             setCurrentPage(currentPage - 1)
//           }
//         >
//           Previous
//         </button>

//         <span>
//           Page {currentPage} of {totalPages}
//         </span>

//         <button
//           disabled={currentPage === totalPages}
//           onClick={() =>
//             setCurrentPage(currentPage + 1)
//           }
//         >
//           Next
//         </button>
//       </div>
//     </div>
//   );
// }


import React, { useState } from "react";
import styles from "./ProductPage.module.css";
import ProductCard from "../components/product/ProductCard";
import EmbeddingRecommendation from "../components/product/dynamicallyRender/EmbeddingRecommendation";
import CollaborativeRecommendation from "../components/product/dynamicallyRender/CollaborativeRecommendation";
import HybridRecommendation from "../components/product/dynamicallyRender/HybridRecommendation";

const ProductPage = () => {
  const [model, setModel] = useState("embedding");
  const [query, setQuery] = useState("");
  
  // Example random products
  const products = [
    { id: 1, name: "Smartphone X", category: "Electronics", description: "Latest AI-powered phone", score: 4.5 },
    { id: 2, name: "Gaming Laptop", category: "Computers", description: "High performance laptop", score: 4.7 },
    { id: 3, name: "Wireless Headphones", category: "Audio", description: "Noise cancelling", score: 4.3 },
  ];

  const renderRecommendation = () => {
    switch (model) {
      case "embedding":
        return <EmbeddingRecommendation query={query} />;
      case "collaborative":
        return <CollaborativeRecommendation query={query} />;
      case "hybrid":
        return <HybridRecommendation query={query} />;
      default:
        return null;
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Recommendation System</h1>
        <div className={styles.controls}>
          <select value={model} onChange={(e) => setModel(e.target.value)} className={styles.dropdown}>
            <option value="embedding">Embedding Model</option>
            <option value="collaborative">Collaborative Model</option>
            <option value="hybrid">Hybrid Model</option>
          </select>
          <input
            type="text"
            placeholder="Search products..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className={styles.searchBar}
          />
        </div>
      </header>

      {renderRecommendation()}

      <section className={styles.products}>
        <h2>Random Products</h2>
        <div className={styles.grid}>
          {products.map((p) => (
            <ProductCard key={p.id} product={p} />
          ))}
        </div>
      </section>
    </div>
  );
};

export default ProductPage;
