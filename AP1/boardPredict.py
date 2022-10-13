import pandas as pd
import re
# import warnings

# warnings.simplefilter("ignore")
data = pd.read_csv("ptt_titles.csv")

def train_predict(l, text, select_model):
    # l = ['寶可夢版', '美劇版','笑話板','軟工版']
    # l = nlpapp.options
    
    # replace
    l = list(map(lambda x: x.replace('八卦版', 'gossiping'), l))
    l = list(map(lambda x: x.replace('寶可夢版', 'pokemon'), l))
    l = list(map(lambda x: x.replace('美劇版', 'EASeries'), l))
    l = list(map(lambda x: x.replace('笑話板', 'joke'), l))
    l = list(map(lambda x: x.replace('軟工版', 'Soft_Job'), l))

    mask = data['Board'].isin(l) # l=['gossiping','joke']
    df = data[mask]

    X = df["Title"]
    y = df["Board"]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # creating a list for appending the preprocessed text
    data_list = []
    # iterating through all the text
    for text in X:
        # removing the symbols and numbers
            text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
            text = re.sub(r'[[]]', ' ', text)
            # converting the text to lower case
            text = text.lower()
            # appending to data_list
            data_list.append(text)

    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(data_list).toarray()
    X.shape # (17166, 19831)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.20)
    if (select_model=='XGBoost'):
        from xgboost import XGBClassifier
        xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)
        xgboostModel.fit(x_train, y_train)
        
        # def predict(text):
        x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
        board = xgboostModel.predict(x) # predicting the board
        board = le.inverse_transform(board) # finding the board corresponding the the predicted value
    else:
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
        model.fit(x_train, y_train)

        # def predict(text):
        x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
        board = model.predict(x) # predicting the board
        board = le.inverse_transform(board) # finding the board corresponding the the predicted value
    # print("這段文字的風格類似",board[0]) # printing the board

    return(board[0])