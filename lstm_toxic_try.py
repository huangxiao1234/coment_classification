from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

"""
使用LSTM模型进行情感分析
"""
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
X_train = train_data["comment_text"].fillna("fillna").values
y_train = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test_data["comment_text"].fillna("fillna").values

#关于调参：整个句子要控制maxlen，原本的数据中maxlen达到了上千，但很多对分类都没有帮助，只需要选取靠前的百来个向量即可
#max_features同样也要控制

max_features = 50000 #最大的样本数
maxlen = 150 #输入句子的最大单词书
EMBEDDING_SIZE = 300
HIDDEN_LAYER_SIZE = 80
BATCH_SIZE = 32
NUM_EPOCHS = 2

tokenizer = Tokenizer(num_words=max_features)#关于Tokenizer的用法自行百度
tokenizer.fit_on_texts(list(X_train) + list(X_test))#需要将训练和测试集的文本加在一起进行词库生成，不然效果很差
X_train = tokenizer.texts_to_sequences(X_train)#将文本转成序列
X_test = tokenizer.texts_to_sequences(X_test)
x_train = pad_sequences(X_train, maxlen=maxlen)#截取到Maxlen长度的序列便于训练
x_test = pad_sequences(X_test, maxlen=maxlen)


#keras的序贯模型
#有两种训练模型的方案
#第一种：对于每个文本，输出所有的特征值作为结果
#第二种：每次只用一个特征作为labels，因此需要训练出多个模型
#在这里采用的是第一种方法
model = Sequential()
model.add(Embedding(max_features, EMBEDDING_SIZE,input_length=maxlen))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE)

#存储模型
model.save_weights('LSTM_weights6.h5')
json_string = model.to_json()
f=open('LSTM6.json','w')
f.write(json_string)
f.close()

#LSTM4:max_feature=30000