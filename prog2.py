import pandas as pd from keras preprocessing.text import Tokenizer from keras preprocessing sequence import pad_sequences from
keras models import Sequential from keras. layers
import Dense, Embedding, LSTM, SpatialDropout1D
from
matplotlib import exhlot from
sklearn model
selection import train_test_split
from keras. utils.p utils import to_categorical import re
from sklearn-preprocessing
import LabelEncoder
from
keras models
import Sequential, load model
import numpx
data = datali texy, sca/sment ent. cs.)
data[ 'text'] = data[ 'text'].apply(lambda x: x. lower ())
data[ 'text'] = data[ 'text'] apply((lambda x: re.sub('[^a-zA-z0-9|s]', "*, *)))
for idx, row in data.iterrows():
row[e] = row[01.replace(rt','')
for idx, row in data. iterrows():
row[0] = row[0j.replace('rt','')
max fatures = 2000
tokenizer = Tokenizer (num words-max_fatures, split=' •) tokenizer. fit_on_texts(data['text'].values)
x = tokenizer. texts_to_sequences(data[ 'text'].values)
x = pad_sequences(X)
labelencoder = LabelEncoder ()
integer_encoded = labelencoder. fit_transform(data['sentiment'])
y - to_categorical (integer_encoded)
x_train, _test, Y_train, Y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)
embed_dim = 128
def createmodel():
model = Sequential()
model. add (Embedding(max_fatures, embed_dim, input_ length = X. shape[1]))
model. add (LSTM(Istm_out, dropout=0.2, recurrent_dropout=0.2))
model. add (Dense(3, activation=' softmax'))
model.compile(loss =
'categorical_crossentropy', optimizer='adam" ,metrics = ['accuracy' ])
return more
batch_size = 32
model = createmodel ()
model. fit(X_train, Y_train, epochs = 1, batch_size=batch_size, verbose = 2)
score, acc = model evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
print (score)
print(acc)
print(model metrics_names)
print(x_train.shape,Y_train.shape)
print(X_test. shape, Y_test.shape)
model = KerasClassifier (build_fn=createmodel, verbose=0)
epochs = |1, 2J
param_grid= dict(epochs=epochs)
grid = GridSearchC(estimator=model, param_grid=param _grid, n_jobs=1)
grid_result= grid.fit(X_train, Y_train,batch_size=32)
print("Best: %f using %" % (grid_result.best_score, grid result.best_params))