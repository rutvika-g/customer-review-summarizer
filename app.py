import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration, pipeline
import streamlit as st
import gc
from models.summarizer import summarize_review

# Streamlit Web App
st.title("Customer Feedback Analysis Tool")
st.write("Analyze customer reviews and view sentiment trends!")

# Sentiment analysis model (Lazy Loading)
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = None
sentiment_model = None
sentiment_pipeline = None

# Summarization model (Lazy Loading)
summarization_model_name = "sshleifer/distilbart-cnn-12-6"
summarization_tokenizer = None
summarization_model = None
summarization_pipeline = None

# Load models only once
def load_sentiment_model():
    global sentiment_tokenizer, sentiment_model, sentiment_pipeline
    if sentiment_pipeline is None:  # Load only if not already loaded
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

def load_summarization_model():
    global summarization_tokenizer, summarization_model, summarization_pipeline
    if summarization_pipeline is None:  # Load only if not already loaded
        summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
        summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)
        summarization_pipeline = pipeline("summarization", model=summarization_model, tokenizer=summarization_tokenizer)

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])

if uploaded_file:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    if "review" not in data.columns:
        st.error("CSV must contain a 'review' column!")
    else:
        # Analyze Sentiments and Summarize Reviews
        sentiments = {"POSITIVE": 0, "NEGATIVE": 0}
        results = []

        # Load models only once
        load_sentiment_model()
        load_summarization_model()

        st.write("### Analysis Results:")
        for review in data['review']:
            # Sentiment Analysis
            sentiment_result = sentiment_pipeline(review)[0]
            sentiment = sentiment_result['label']
            sentiments[sentiment] += 1

            # Summarize the review
            summary = summarize_review(review, summarization_pipeline)

            results.append({
                "Review": review,
                "Sentiment": sentiment,
                "Confidence": f"{sentiment_result['score']:.2f}",
                "Summary": summary
            })

        # Display results in a table
        results_df = pd.DataFrame(results)
        st.write(results_df)

        # Plot sentiment distribution
        st.write("### Sentiment Distribution:")
        fig, ax = plt.subplots()
        ax.bar(sentiments.keys(), sentiments.values(), color=['green', 'red'])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        # Export results to Excel
        st.write("### Download Results:")
        output_file = "feedback_results.xlsx"
        results_df.to_excel(output_file, index=False)

        with open(output_file, "rb") as file:
            st.download_button(
                label="Download as Excel",
                data=file,
                file_name="feedback_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Clear models and force garbage collection after use
        del sentiment_pipeline, summarization_pipeline
        gc.collect()  # Clean up memory