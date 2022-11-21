import re
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
import textwrap, json, joblib
from sklearn.naive_bayes import MultinomialNB
import os
#
def preprocess_function(file_name,content_type=None):

    text = file_name.readlines()
    return text[0]


def predict_function(input_data, model):
    vectorizer, model = model
    data = vectorizer.transform([input_data]).toarray()
    result = list(model.predict(data))[0]
    return result


def model_load_function(model_file_path):
    token_path = os.path.join(model_file_path, "transform.pkl")
    model_path = os.path.join(model_file_path, "model.pkl")
    # load the model
    cv = joblib.load(token_path)
    model = joblib.load(model_path)
    return cv, model


def postprocess_function(predictions,content_type=None):
    return json.dumps({"response": "Text detected : {}".format(predictions)})

# test the script
"""
if __name__ == "__main__":
    file_name = "./Data/sample.txt"
    model_file_path = "./model_files"
    #vector_file_path = "./model_files/language_detection_vectorizer.pickle"
    model = model_load_function(model_file_path)
    input_data = preprocess_function(file_name)
    summary = predict_function(input_data, model)
    output = postprocess_function(summary)
    print(output)
    
"""
