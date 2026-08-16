# Aim: to find best stocks to buy and sell in the stock market using AI and ML algorithms. The program will analyze historical data, news sentiment, and other relevant factors to make informed decisions.

from manager import Manager

with Manager(
    model="gemma4:12b",          # Manager's own model (coordination & synthesis)
    num_workers=6,               # Spawn 6 Qwen workers
    default_timeout_seconds=60,
    max_timeout_seconds=1800,
    num_ctx=32768,
    max_parallel=6,
) as mgr:
    
    # Override each worker's model to Qwen 3 (4B) — lightweight & fast
    for worker in mgr.workers:
        worker.model = "qwen3:4b"

    print("=" * 60)
    print("MANAGER: gemma4:12b  |  WORKERS: qwen3:4b x6")
    print("=" * 60)
    
    print("\n--- 5. SPLIT-AND-CONQUER ---")
    complex_task = (
        """
        To design a comprehensive stock market analysis system, we need to break down the task into several sub-tasks:
        1. Data Collection: Gather historical stock prices, trading volumes, and other relevant financial data from various sources.
        2. Data Preprocessing: Clean and normalize the collected data, handle missing values, and prepare it for analysis.
        3. Feature Engineering: Create meaningful features from the raw data, such as moving averages, volatility measures, and sentiment scores from news articles.
        4. Model Development: Train machine learning models to predict stock price movements based on the engineered features.
        5. Backtesting: Evaluate the performance of the trained models using historical data to ensure their effectiveness.
        6. Strategy Optimization: Fine-tune the trading strategies based on backtesting results to maximize returns and minimize risks.
        7. Storage: The model and all files related to the stock market analysis system should be stored in a folder in my desktop for easy access and future reference.
        8. Documentation: Create comprehensive documentation for the system, including setup instructions, usage guidelines, and explanations of the underlying algorithms and models.
        """
    )
    design_doc = mgr.split_and_conquer(
        complex_query=complex_task,
        n_subtasks=6,  # Gemma breaks this into 6 parallel sub-tasks
    )
    print(f"Split-and-Conquer Result:\n{design_doc}")