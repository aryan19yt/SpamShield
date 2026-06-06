import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')

ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    words=word_tokenize(text)
    words=[word for word in words if word.isalnum()]
    stop_words=set(stopwords.words('english'))
    words=[word for word in words if word not in stop_words]
    words=[ps.stem(word) for word in words]
    return " ".join(words)

df=pd.read_csv("data/spam.csv", encoding="latin-1")
df=df[['v1', 'v2']]
df.columns=['label', 'message']

df['label']=df['label'].map({
    'ham':0,
    'spam':1
})

df['message']=df['message'].apply(transform_text)

X=df['message']
y=df['label']

vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model=MultinomialNB()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and Vectorizer saved successfully!")