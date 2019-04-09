import tensorflow as tf

mnist =tf.keras.datasets.mnist

(x_train,y_train) , (x_test,y_test) = n=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10	, activation=tf.nn.softmax))

model.compile (optimizer='adam',
				loss='sparse_categorical_crossentropy',	
				metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)


import matplotlib.pyplot as plt 
plt.imshow(x_train[0],cmap=plt.cm.binary)

model.save_model('number_predict')
new_model =tf.keras.models.load_model('number_predict')
predictions=new_model.predict([x_test])
print(predictions)

import numpy as np
print(np.argmax(prediction[0]))
plt.imshow(x_train[0])
plt.show()


print (x_train[0])