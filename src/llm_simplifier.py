from transformers import pipeline

def simplify_math_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=60, min_length=15, do_sample=False)
    return summary[0]['summary_text']
