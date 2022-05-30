import streamlit as st
import numpy as np
from prediction import clean_text,lemma_clean_text,encode_docs,predict_sentiment
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

max_length=262

def get_token():
    with open('Model/token.pickle', 'rb') as handle:
        token = pickle.load(handle)
        return token

st.set_page_config(page_title="Toxicity Prediction ", layout="centered")
st.markdown("<h1 style='text-align: center;'>Toxicity Prediction</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):
        input_text=st.text_input("Enter the text", 'input text')
        submit = st.form_submit_button("Predict")

        if submit:
            token=get_token()
            model=load_model('Model/nlpModel.h5')

            cl_text=clean_text(input_text)
            lem_text=lemma_clean_text(cl_text)
            pad_text=encode_docs(token, max_length,lem_text)
            result=predict_sentiment(pad_text,model)
            print(result)
            st.write(f"The predicted toxicity is: {result}")     





if __name__ == '__main__':
    main()
