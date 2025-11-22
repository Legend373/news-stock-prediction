from src.market_data_processor import MarketDataProcessor

def main():
    file_path = "data/stock_prices.csv"  # Adjust as needed

    processor = MarketDataProcessor(file_path)

    print("Loading data...")
    processor.load_data()

    print("Applying TA-Lib indicators...")
    processor.apply_ta_indicators()

    print("Calculating financial metrics...")
    processor.calculate_financial_metrics()

    print("Creating visualizations...")
    processor.visualize(save_path="outputs/price_analysis.png")

    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()
