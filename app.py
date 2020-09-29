import streamlit as st
import backend

st.write("""
# Text Sentiment Prediction App
## Enter text to see its sentiment, predicted by a machine learning model trained on Twitter sentiment data
Find the GitHub repo including the deep learning model training/testing [here](https://github.com/LowkeyBBQ/sentiment-web-app)
""")

user_input = st.text_input("Enter your text here", 'The cake is a lie')
prob = backend.predict(user_input)[0]
sentiment = "Positive" if prob >0.5 else "Negative"
st.write("Predicted sentiment: {}, Probability(Positive) = {}".format(sentiment, prob))