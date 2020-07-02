#Deepank G
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter("ignore")

df = pd.read_csv('results.csv')

df['text'] = df['text'].str.lower()
df['text'].replace(regex=True, inplace=True, to_replace=r'@[a-z0-9_.]+', value=r'')
df['text'].replace(regex=True, inplace=True, to_replace=r'\w+.\/\/\S+', value=r'')
df['text'].replace(regex=True, inplace=True, to_replace=r'[^a-z|A-Z|^\s]', value=r'')
df['sentiment'] = df['sentiment'].replace(['negative', 'positive', 'neutral'], [0, 1, 2])
df['sentiment'] = df['sentiment'].astype('category')

loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
y_pred = loaded_model.predict(df['text'])
y_true = df['sentiment']
df['predicted'] = y_pred

df.to_csv("predictions.csv")
result = accuracy_score(y_true, y_pred)
print("\nAccuracy: ",result)
