def summarize_review(review, summarization_pipeline):
    summary = summarization_pipeline(review, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']
