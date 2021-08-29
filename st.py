import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale= 2)

import re
import string
import os
from tqdm import tqdm
from time import time


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets, models
from PIL import Image

from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report

model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
model_path = os.path.join('/', model_name)
vect_path = os.path.join('/', vectorizer_name)

model=pickle.dump(clf, open(model_name, 'wb'))## Save model
vectorizer=pickle.dump(vectorizer, open(vectorizer_name, 'wb'))## Save tfidf-vectorizer
loaded_model =pickle.load(open('rf_model.pk', 'rb'))
loaded_vect = pickle.load(open('tfidf_vectorizer.pk', 'rb'))

def cleaning (review):
    review_c=[]
    review_1=word_tokenize(review)
    for i in review_1:
        if i.lower() not in stop_words:
            re_review=re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", i)
            review_s=SnowballStemmer('english').stem(re_review)
            review_c.append(review_s)
    return " ".join(review_c)
    
    
    from sklearn.feature_extraction.text import TfidfVectorizer
def raw_test(review, model, vectorizer):
    # Clean Review
    review_c =cleaning(review)
    # Embed review using tf-idf vectorizer
    embedding =vectorizer.transform([review_c])
    # Predict using your model
    prediction =model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"
    
    def main():
    st.title('Predict the score of review")
    st.markdown("<h1 style='text-align: center; color: White;background-color:#0E1117'>Food Review Classifier</h1>", unsafe_allow_html=True)
    review = st.text_input(label='Write your Review')
    if st.button('Classify'):
        result = raw_test(review, model, vect)
        st.success(
            'This Review Is {}'.format(result))


if __name__ == '__main__':
    main()
