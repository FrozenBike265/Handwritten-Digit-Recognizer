#probleme importuri !!!!!!!!!!!!!!!!!!!!!!!

import os
# Setam 2 variabile de mediu pentru performanta TensorFlow-ului si 
# pentru suprimarea unor log-uri
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Dezactiveaza unele optimizari
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Reduce log-urile TensorFlow

import tensorflow as tf
from tensorflow import keras

#from tensorflow.keras.models import Sequential
#from keras.datasets import mnist
#from keras.models import Sequential
#from keras.layers import Dense, Flatten
#from keras.layers import Dropout
#from keras.layers import Flatten
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D

# Limitam thread-urile folosite pentru a optimiza utilizarea CPU-ului si GPU-ului
os.environ["OMP_NUM_THREADS"] = "8" 
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"  
os.environ["TF_NUM_INTEROP_THREADS"] = "4" 

# Importam setul de date MNIST si layerele necesare pentru CNN
mnist = tf.keras.datasets.mnist # MNIST 
sequential = tf.keras.models.Sequential # Sequential 
dense = tf.keras.layers.Dense # Dense layer
flatten = tf.keras.layers.Flatten # Flatten layer
dropout = tf.keras.layers.Dropout # Dropout layer
conv2D = tf.keras.layers.Conv2D # Convolutional layer
maxPooling2D = tf.keras.layers.MaxPooling2D # Max pooling layer
batchMario = tf.keras.layers.BatchNormalization

# Incarcam setul de date MNIST si il impartim in 2 seturi de date:
# unul unde vom da train si unul unde vom testa
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# --------------------------- Prelucrarea datelor ---------------------------

num_classes = 10 # Numarul de clase -> (cifre de la 0 la 9)

# Redimensionam datele de intrare -> adaugam inca o dimensiune pentru imaginile negre
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Convertim clasele in formatul claselor, o data pentru train si o data pentru test
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Normalizam valorile pixelilor la intervalul [0,1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# --------------------------- Crearea modelului ---------------------------

batch_size = 128 # Marimea unui batch
num_classes = 10 # Am pus din nou nr clase
epochs = 150 # Numarul de grupari pentru antrenarea modelului CNN


# Prima imbunatatire:
# Adaugam mai multe straturi convolutionale cu filtre de dimensiuni diferite.

model = sequential()
model.add(conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(batchMario())
model.add(conv2D(32, (3, 3), activation='relu'))
model.add(batchMario())
model.add(maxPooling2D(pool_size=(2, 2)))
model.add(dropout(0.3))

model.add(conv2D(64, (3, 3), activation='relu'))
model.add(batchMario())
model.add(conv2D(64, (3, 3), activation='relu'))
model.add(batchMario())
model.add(maxPooling2D(pool_size=(2, 2)))
model.add(dropout(0.4))

model.add(flatten())
model.add(dense(512, activation='relu'))
model.add(dropout(0.5))
model.add(dense(num_classes, activation='softmax'))

# A doua imbunatatire
# Compilam modelul cu optimizerul Adam
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

# --------------------------- Antrenarea Modelului ---------------------------

# Antrenam modelul pe train data
# verbose = 1 ->  Arata progresul in timpul training-ului
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("Modelul a fost invatat cu succes!")
model.save('mnist.h5')

# --------------------------- Evaluarea modelului ---------------------------

# Evaluarea modelului pe test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Loss: ', score[0]) # Pierderea pe test data
print('Accuracy: ', score[1]) # Acuratetea pe test data
