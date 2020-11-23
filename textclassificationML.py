import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import regex as re
import nltk
import h2o
from nltk.stem.porter import PorterStemmer
from h2o.automl import H2OAutoML

h2o.init()

#we download punkt which is used as a sentence tokenizer

#lets read in the csv file

df = pd.read_csv(r"C:\Users\aniru\Desktop\complaints.csv")

#analyze the data
df['Product'].value_counts()
df['Product'].value_counts().plot(kind='bar')

df['Company'].value_counts()


complaints_df = df[['Consumer complaint narrative','Product','Company']].rename(columns={'Consumer complaint narrative':'complaints'})
pd.set_option('display.max_colwidth',-1)

target = {'Debt collection':0, 'Credit card or prepaid card':1, 'Mortage':2, 'Checking or savings account':3, 'Student loan':4, 'Vehicle loan or lease':5}
complaints_df['target'] = complaints_df['Product'].map(target)

#create training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(complaints_df, test_size=0.8, random_state=111)

#lets implement sentence tokenization

stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(nltk.corpus.stopwords.words('english'))
def tokenize(text):
    tokens = [word for word in nltk.word_tokenize((text)) if (len(word) > 3 and len(word.strip('Xx/'))>2 and len(re.sub('\d+', '', word.strip('Xx/'))) >3)]
    stems = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    return stems

#the sentence tokenization should decompose the words into tokens based on the
#conditional statements as above. once that has been done we can train our model as follows

vectorizer_tf = TfidfVectorizer(tokenizer=tokenize, stop_words=None, max_df=0.75, max_features=1000, lowercase=False, ngram_range=(1,2))
train_vectors = vectorizer_tf.fit_transform(X_train.complaints)

#training our model we get



vectorizer_tf.get_feature_names()

#test data

test_vectors = vectorizer_tf.transform(X_test.complaints)

#observing the shape of the train data we get


#we now create a dataframe with the vectorized features as the columns
#and also concatenate it with that of x_train to get a complete dataframe

train_df=pd.DataFrame(train_vectors.toarray(), columns=vectorizer_tf.get_feature_names())
train_df=pd.concat([train_df,X_train['target'].reset_index(drop=True)], axis=1)

#we do the same for the test data and create test_df

test_df=pd.DataFrame(test_vectors.toarray(), columns=vectorizer_tf.get_feature_names())
test_df=pd.concat([test_df,X_test['target'].reset_index(drop=True)], axis=1)

#we apply conversion of the above pandas dataframes to that of H2O
#dataframe objects as below

h2o_train_df = h2o.H2OFrame(train_df)
h2o_test_df = h2o.H2OFrame(test_df)



h2o_train_df['target'] = h2o_train_df['target'].asfactor()
h2o_test_df['target'] = h2o_test_df['target'].asfactor()

#applying AutoML

aml = H2OAutoML(max_models = 10, seed = 10, exclude_algos = ["StackedEnsemble"], verbosity="info", nfolds=0, balance_classes=True, max_after_balance_size=0.3)


x=vectorizer_tf.get_feature_names()
y='target'

#training the automl model with respect to train_df

aml.train(x = x, y = y, training_frame = h2o_train_df, validation_frame=h2o_test_df)

#observe performance of the aml model
aml.leaderboard()


#predicting the aml model with test dataset

pred=aml.leader.predict(h2o_test_df)

#observing performance of the prediction
aml.leader.model_performance(h2o_test_df)

model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
out = h2o.get_model([mid for mid in model_ids if "XGBoost" in mid][0])

out.convert_H2OXGBoostParams_2_XGBoostParams()

#importing XG boost



from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb


xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, objective='multi:softprob', random_state=10, **{"updater": "grow_gpu"})
predictions = xgb_clf.predict(test_vectors)
cm = confusion_matrix(X_test['target'], predictions)
print(cm)

#get classification report for the target feature


print('classification_report :\n',classification_report(X_test['target'], predictions))

from sklearn.utils import class_weight
class_weights = list(class_weight.compute_class_weight('balanced',
                                             np.unique(X_train['target']),
                                             X_train['target']))


weights = np.ones(X_train.shape[0], dtype = 'float')

for i, val in enumerate(X_train['target']):
    weights[i] = class_weights[val]

xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, objective='multi:softprob', random_state=10, **{"updater": "grow_gpu"})

xgb_clf.fit(train_vectors, X_train['target'], sample_weight=weights)

predictions = xgb_clf.predict(test_vectors)

cm = confusion_matrix(X_test['target'], predictions)
print(cm)

cm = confusion_matrix(X_test['target'], predictions)
print(cm)