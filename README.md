# Customer Feedback Analysis Tool

This project allows you to analyze customer feedback, categorize sentiments (positive/negative), summarize reviews, and visualize sentiment trends using a Streamlit app.

## Features

- **Sentiment Analysis**: Classifies reviews into positive or negative sentiments.
- **Review Summarization**: Summarizes customer reviews using a pre-trained summarization model.
- **Visualization**: Displays a bar chart of sentiment distribution.
- **Download Results**: Export the results to an Excel file for further analysis.

## Tech Stack

- **Streamlit**: For creating the interactive web app.
- **Hugging Face Transformers**: To use pre-trained sentiment analysis and summarization models.
- **Pandas**: For handling data.
- **Matplotlib**: For visualizing sentiment distribution.
- **Openpyxl**: For exporting the results to Excel.

## How to Run

1. Clone or download the project files.
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the app using Streamlit:
    ```bash
    streamlit run app.py
    ```
4. Upload a CSV file with customer reviews. Make sure the file has a `review` column.
5. View sentiment analysis, review summaries, and download the results.

## My Aspirations

I’m interested in pursuing a major in Business and am passionate about using technology to drive business insights. I love combining technical skills with business strategies, and I hope to use this project as a step toward building more advanced solutions in the future!


## Live Demo

You can view the live demo of the Customer Review Summarizer app [here](https://customer-review-summarizer-rutvi.streamlit.app/).

## How to Use the App

1. **Access the app**: Visit the live demo link above or run the app locally.
2. **Upload your reviews file**: The app works with a CSV file containing customer reviews. You can use the `data/reviews.csv` file provided in the repository.

### Example Data

You can try the app with the example data under `data/reviews.csv`. The data in this file contains sample customer reviews that will be processed and summarized.
