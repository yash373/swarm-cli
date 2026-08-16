from worker import Worker

worker1 = Worker("qwen3:8b")
worker1.respond(
    """
    Create a mathematical model that predicts who is going to win the 2026 F1 World Championship based on the performance of the drivers in the 2023, 2024, and 2025 seasons. Use the data from the last three seasons to make your predictions. You can use any publicly available data sources or APIs to gather information about the drivers' performance, such as their race results, qualifying positions, and points standings. You can also use any Python packages or libraries that you think will be helpful in creating your model. Please provide a detailed explanation of your methodology and the results of your predictions.
    Create a report that includes your model, the data you used, and your predictions. The report should be in a format that can be easily shared with others, such as a PDF or a Jupyter notebook. Please include any visualizations or charts that help to illustrate your findings.
    """
)