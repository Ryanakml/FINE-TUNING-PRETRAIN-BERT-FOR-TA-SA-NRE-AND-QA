# BERT Model Demonstration App

This Streamlit application demonstrates various capabilities of BERT (Bidirectional Encoder Representations from Transformers) models for natural language processing tasks.

## Features

The application showcases several key BERT functionalities:

- **Sentiment Analysis**: Determine whether text is positive, negative, or neutral
- **Named Entity Recognition**: Identify entities such as persons, organizations, and locations in text
- **Question Answering**: Extract answers to questions from a given context
- **Text Classification**: Categorize text into different emotions
- **About BERT**: Learn about BERT's architecture and how it works

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app with:

```bash
streamlit run app.py
```

The application will open in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Requirements

This application requires:
- Python 3.7+
- Streamlit
- Transformers (Hugging Face)
- PyTorch
- Pandas
- Matplotlib
- Seaborn
- NumPy

Detailed version requirements are specified in the `requirements.txt` file.

## How It Works

The application uses pre-trained BERT models from the Hugging Face Transformers library to perform various NLP tasks. The models are loaded on demand when a specific task is selected, and the results are displayed with visualizations to help understand the model's output.

## Learning Outcomes

By exploring this application, you can learn:

1. How BERT processes and understands text
2. The versatility of BERT for different NLP tasks
3. How to implement BERT models using the Transformers library
4. How to create interactive NLP applications with Streamlit

## Future Improvements

Potential enhancements for this application:

- Add more BERT-based tasks such as text summarization
- Allow users to upload their own fine-tuned BERT models
- Implement comparison between BERT and other transformer models
- Add multilingual support