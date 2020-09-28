import tensorflow as tf
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

import json
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

model = tf.keras.models.load_model("model.h5")
MAX_SEQUENCE_LENGTH = 30
def predict(text):
  return model.predict(tf.keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences([preprocess(text)]), maxlen=MAX_SEQUENCE_LENGTH), verbose=1)
