import pickle

filename = "language_predictor.pkl"
loaded_model = pickle.load(open(filename, 'rb'))
encodename = "encoder.pkl"
le = pickle.load(open(encodename, 'rb'))
count_vect_name = "vectorize.pkl"
cv = pickle.load(open(count_vect_name, 'rb'))

def load_predict(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = loaded_model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     return lang[0]