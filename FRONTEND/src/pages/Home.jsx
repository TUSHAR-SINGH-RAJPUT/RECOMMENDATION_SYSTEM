import Style from "./Home.module.css";

export default function HomePage() {

  function open_product_page() {
    
  }

  return (
    <div className={Style.HomePage}>

      <h1 >Welcome to the Home Page</h1>

      <div className={Style.buttonContainer}>

        <button className="universalBtn" onClick={open_product_page}>
          Explore
        </button>

        <button
          className="universalBtn"
          onClick={open_product_page}
        >
          Whats New
        </button>

      </div>

    </div>
  );
}