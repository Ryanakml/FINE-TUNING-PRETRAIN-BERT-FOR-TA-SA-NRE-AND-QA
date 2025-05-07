import streamlit as st
import torch
import gc
from transformers import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering, BertForTokenClassification, pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set memory management for PyTorch
torch.set_num_threads(1)  # Limit number of threads to prevent memory issues

# Set page configuration
st.set_page_config(page_title="BERT Demo App", page_icon="🤖", layout="wide")

# App title and description
st.title("BERT Model Demonstration")
st.markdown("""
This application demonstrates various capabilities of BERT (Bidirectional Encoder Representations from Transformers).
Explore different NLP tasks powered by BERT models.
""")

# Sidebar for navigation
st.sidebar.title("BERT Capabilities")
task = st.sidebar.radio(
    "Select a task:",
    ["Sentiment Analysis", "Named Entity Recognition", "Question Answering", "Text Classification", "About BERT"]
)

# Load models based on selected task
@st.cache_resource
def load_sentiment_model():
    # Explicitly set device to CPU to avoid bus error
    # Set small batch size and optimize memory usage
    return pipeline("sentiment-analysis", device="cpu", batch_size=1, framework="pt")

@st.cache_resource
def load_ner_model():
    # Explicitly set device to CPU to avoid bus error
    # Set small batch size and optimize memory usage
    return pipeline("ner", device="cpu", batch_size=1, framework="pt")

@st.cache_resource
def load_qa_model():
    # Explicitly set device to CPU to avoid bus error
    # Set small batch size and optimize memory usage
    return pipeline("question-answering", device="cpu", batch_size=1, framework="pt")

@st.cache_resource
def load_classification_model():
    # Using a pre-trained model fine-tuned on emotion classification
    # Explicitly set device to CPU to avoid bus error
    # Set small batch size and optimize memory usage
    return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", device="cpu", batch_size=1, framework="pt")

# About BERT section
if task == "About BERT":
    st.header("About BERT")
    st.markdown("""
    ## Bidirectional Encoder Representations from Transformers
    
    BERT is a transformer-based machine learning technique for natural language processing (NLP) pre-training 
    developed by Google. It was created and published in 2018 by Jacob Devlin and his colleagues from Google.
    
    ### Key Features:
    
    - **Bidirectional Training**: BERT reads text input in both directions simultaneously, allowing it to understand the context of a word based on all of its surroundings.
    
    - **Pre-training & Fine-tuning**: BERT is pre-trained on a large corpus of unlabeled text including the entire Wikipedia and Book Corpus. It can then be fine-tuned for specific tasks.
    
    - **State-of-the-art Performance**: BERT achieved state-of-the-art results on a wide range of NLP tasks when it was released.
    
    ### How BERT Works:
    
    BERT uses a transformer architecture that processes words in relation to all other words in a sentence, rather than one-by-one in order. This allows the model to learn context from all surroundings.
    
    During pre-training, BERT uses two strategies:
    
    1. **Masked Language Modeling (MLM)**: Some percentage of input tokens are masked, and the model attempts to predict them based on context.
    
    2. **Next Sentence Prediction (NSP)**: The model receives pairs of sentences and learns to predict if the second sentence follows the first in the original document.
    
    This application demonstrates some of the practical applications of BERT models in various NLP tasks.
    """)
    
    # Display BERT architecture visualization
    st.image("https://miro.medium.com/max/1400/1*eTwQsyTiF-jZSaYUTsP7gQ.png", caption="BERT Architecture", use_column_width=True)

# Sentiment Analysis section
elif task == "Sentiment Analysis":
    st.header("Sentiment Analysis with BERT")
    st.markdown("""
    Sentiment analysis determines whether a piece of text is positive, negative, or neutral.
    BERT excels at understanding context, making it powerful for sentiment analysis.
    """)
    
    # User input
    user_input = st.text_area("Enter text for sentiment analysis:", "I love how BERT understands the context of words in a sentence.")
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing..."):
            # Load model and predict
            sentiment_model = load_sentiment_model()
            result = sentiment_model(user_input)
            
            # Clean up memory
            del sentiment_model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Display result
            sentiment = result[0]['label']
            score = result[0]['score']
            
            # Create columns for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Result")
                if "POSITIVE" in sentiment:
                    st.success(f"Sentiment: {sentiment}")
                    emoji = "😊"
                elif "NEGATIVE" in sentiment:
                    st.error(f"Sentiment: {sentiment}")
                    emoji = "😞"
                else:
                    st.info(f"Sentiment: {sentiment}")
                    emoji = "😐"
                
                st.markdown(f"### {emoji} Confidence: {score:.2%}")
            
            with col2:
                st.subheader("Confidence Visualization")
                fig, ax = plt.subplots()
                sentiment_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Negative'],
                    'Score': [score if 'POSITIVE' in sentiment else 1-score, 
                              score if 'NEGATIVE' in sentiment else 1-score]
                })
                sns.barplot(x='Sentiment', y='Score', data=sentiment_data, ax=ax)
                ax.set_ylim(0, 1)
                st.pyplot(fig)

# Named Entity Recognition section
elif task == "Named Entity Recognition":
    st.header("Named Entity Recognition with BERT")
    st.markdown("""
    Named Entity Recognition (NER) identifies entities such as persons, organizations, locations, etc. in text.
    BERT's contextual understanding helps it accurately identify and classify named entities.
    """)
    
    # User input
    user_input = st.text_area("Enter text for entity recognition:", "Google CEO Sundar Pichai announced new AI features at the conference in New York last week.")
    
    if st.button("Identify Entities"):
        with st.spinner("Processing..."):
            # Load model and predict
            ner_model = load_ner_model()
            results = ner_model(user_input)
            
            # Clean up memory
            del ner_model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Process results
            entities = {}
            for result in results:
                entity_type = result['entity']
                if entity_type.startswith('B-') or entity_type.startswith('I-'):
                    entity_type = entity_type[2:]  # Remove B- or I- prefix
                
                word = result['word']
                if word.startswith('##'):
                    word = word[2:]  # Remove ## prefix for subwords
                
                if entity_type in entities:
                    if word.startswith('##'):
                        entities[entity_type][-1] += word
                    else:
                        entities[entity_type].append(word)
                else:
                    entities[entity_type] = [word]
            
            # Display results
            if entities:
                st.subheader("Identified Entities")
                
                # Create a DataFrame for better visualization
                entity_data = []
                for entity_type, words in entities.items():
                    for word in words:
                        entity_data.append({
                            'Entity Type': entity_type,
                            'Text': word,
                        })
                
                entity_df = pd.DataFrame(entity_data)
                st.table(entity_df)
                
                # Highlight entities in the original text
                st.subheader("Highlighted Text")
                highlighted_text = user_input
                for entity_type, words in entities.items():
                    for word in words:
                        if word in highlighted_text:
                            highlighted_text = highlighted_text.replace(
                                word, 
                                f"<span style='background-color: yellow; font-weight: bold;'>{word}</span>"
                            )
                
                st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                st.info("No entities detected in the provided text.")

# Question Answering section
elif task == "Question Answering":
    st.header("Question Answering with BERT")
    st.markdown("""
    BERT can extract answers to questions from a given context. This demonstrates BERT's reading comprehension abilities.
    """)
    
    # User input
    context = st.text_area(
        "Context:", 
        "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google. BERT was pretrained on two tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)."
    )
    
    question = st.text_input("Question:", "Who created BERT?")
    
    if st.button("Answer"):
        if context and question:
            with st.spinner("Finding answer..."):
                # Load model and predict
                qa_model = load_qa_model()
                result = qa_model(question=question, context=context)
                
                # Clean up memory
                del qa_model
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Display result
                answer = result['answer']
                score = result['score']
                
                st.subheader("Answer")
                st.info(answer)
                st.markdown(f"**Confidence:** {score:.2%}")
                
                # Highlight answer in context
                st.subheader("Answer in Context")
                highlighted_context = context.replace(
                    answer, 
                    f"<span style='background-color: #FFFF00; font-weight: bold;'>{answer}</span>"
                )
                st.markdown(highlighted_context, unsafe_allow_html=True)
        else:
            st.warning("Please provide both context and question.")

# Text Classification section
elif task == "Text Classification":
    st.header("Text Classification with BERT")
    st.markdown("""
    This demo uses a BERT model fine-tuned for emotion classification to categorize text into emotions.
    """)
    
    # User input
    user_input = st.text_area("Enter text for emotion classification:", "I'm so excited about learning how BERT works!")
    
    if st.button("Classify Emotion"):
        with st.spinner("Classifying..."):
            # Load model and predict
            classification_model = load_classification_model()
            result = classification_model(user_input)
            
            # Clean up memory
            del classification_model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Display result
            emotion = result[0]['label']
            score = result[0]['score']
            
            # Map emotions to emojis
            emotion_emojis = {
                'joy': '😄',
                'sadness': '😢',
                'anger': '😠',
                'fear': '😨',
                'love': '❤️',
                'surprise': '😲'
            }
            
            emoji = emotion_emojis.get(emotion, '🤔')
            
            # Create columns for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Result")
                st.markdown(f"### Emotion: {emotion.capitalize()} {emoji}")
                st.markdown(f"**Confidence:** {score:.2%}")
            
            with col2:
                st.subheader("Emotion Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create sample data for other emotions with lower values
                emotions = list(emotion_emojis.keys())
                scores = [0.1] * len(emotions)
                
                # Set the detected emotion's score
                emotion_index = emotions.index(emotion)
                scores[emotion_index] = score
                
                # Create bar chart
                bars = sns.barplot(x=emotions, y=scores, ax=ax)
                
                # Add emojis to bars
                for i, e in enumerate(emotions):
                    ax.text(i, scores[i] + 0.02, emotion_emojis[e], ha='center', fontsize=20)
                
                ax.set_ylim(0, 1.1)
                ax.set_xticklabels([e.capitalize() for e in emotions])
                ax.set_ylabel('Confidence Score')
                ax.set_title('Emotion Classification Results')
                
                st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About this app
This app demonstrates various capabilities of BERT models for natural language processing tasks.

Built by Ryan Akmal Pasya 

GitHub: [ryanakmalpasya](GitHub: [ryanakmalpasya](URL_ADDRESS.com/ryanakmalpasya)
""")