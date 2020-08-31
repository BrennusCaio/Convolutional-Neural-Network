#CONSTANTES
Epocas          = 50
batch_size      = 128
NumeroResistor  = 5080
NumeroCapacitor = 5220
NumeroTransistor= 5185
NumeroIC        = 4106

#BIBLIOTECAS
import cv2
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

'exec(%matplotlib inline)'

import os
import random
import gc

#Diretório de Treino e Teste
diretorio_treino = 'C:\\Users\\Brennus\\Downloads\\Dataset_Treino'
diretorio_teste = 'C:\\Users\\Brennus\\Downloads\\Dataset_Teste'


#Coleta imagens de treino com nome de Capacitor, Resistor e Transistor
treino_capacitor = ['C:\\Users\\Brennus\\Downloads\\Dataset_Treino\\{}'.format(i) for i in os.listdir(diretorio_treino) if 'Capacitor' in i]
treino_resistor = ['C:\\Users\\Brennus\\Downloads\\Dataset_Treino\\{}'.format(i) for i in os.listdir(diretorio_treino) if 'Resistor' in i]
treino_transistor = ['C:\\Users\\Brennus\\Downloads\\Dataset_Treino\\{}'.format(i) for i in os.listdir(diretorio_treino) if 'Transistor' in i]
treino_ic = ['C:\\Users\\Brennus\\Downloads\\Dataset_Treino\\{}'.format(i) for i in os.listdir(diretorio_treino) if 'IC' in i]

imagens_treino = treino_capacitor[:NumeroCapacitor] + treino_resistor[:NumeroResistor] + treino_transistor[:NumeroTransistor] + treino_ic[:NumeroIC]

random.shuffle(imagens_treino)


del treino_capacitor
del treino_resistor
del treino_transistor
gc.collect()

numero_colunas = 224
numero_linhas = 224

def processar_imagens(lista_imagens):

    X = [] # images
    y = [] # labels


    for image in lista_imagens:
        try:
            #Redimensiona as Imagens de treino para 150x150 com 3 canais de cores (RGB)
            #Método de interpolação para realizar o zoom da imagem
            X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR), (numero_colunas, numero_linhas), interpolation=cv2.INTER_CUBIC))
            #get the labels
            if 'Capacitor' in image:
                y.append(0)
            elif 'Resistor' in image:
                y.append(1)
            elif 'Transistor' in image:
                y.append(2)
            elif 'IC' in image:
                y.append(3)

        #Caso ele não consiga ler as imagens (cv2.imread) elas vão ser ignoradas.
        except Exception as e:
            print(str(e))
    
    return X, y

#get the train and label data
X, y = processar_imagens(imagens_treino)

#Convert list to numpy array
X = np.array(X)
y = np.array(y)

del imagens_treino
gc.collect()

#train_test_split realiza a divisão do dataset de treino entre: dataset de treino e dataset de validação.
X_treino, X_validacao, y_treino, y_validacao = train_test_split(X, y, test_size=0.20, random_state=2)

del X
del y
gc.collect()

print("Formato(shape) das imagens de treino:", X.shape)
print("Formato(shape) dos rótulos          :", y.shape)
print("Shape of train images is:", X_treino.shape)
print("Shape of validation images is:", X_validacao.shape)
print("Shape of labels is:", y_treino.shape)
print("Shape of labels is:", y_validacao.shape)


ntreino = len(X_treino)
nvalidacao = len(X_validacao)


#Criar modelo
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

sumario_do_modelo = model.summary()

#CALLBACKS
'''
callback1 = ModelCheckpoint(filepath='_melhor_modelo9.hdf5',
                             monitor='val_loss',
                             save_best_only=True)

callback2 = [EarlyStopping(monitor = 'acc', patience = 2)]

callback3 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=0.0001)

meus_callbacks = [callback1, callback2, callback3]
'''

#OTIMIZADOR
opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)


#PREPARANDO A REDE PARA O TREINO
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])

#DATA_AUGMENTION
treino_datagen = ImageDataGenerator(rescale=1. / 255,  #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True, )

validacao_datagen = ImageDataGenerator(rescale=1. / 255)  #We do not augment validation data. we only perform rescale


#IMAGE GENERATORS
treino_generator = treino_datagen.flow(X_treino, y_treino, batch_size=batch_size)
validacao_generator = validacao_datagen.flow(X_validacao, y_validacao, batch_size=batch_size)


#FUNÇÃO DE INICIO DO TREINO
history = model.fit(treino_generator,
                    steps_per_epoch=ntrain // batch_size,
                    epochs=Epocas,
                    validation_data=validacao_generator,
                    validation_steps=nval // batch_size)

#SALVAR MODELO
model.save('modelo.h5')

#METRICAS
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


#GRÁFICO DA ACURÁCIA DE VALIDAÇÃO E TREINO
plt.plot(epochs, acc, 'b', label='Treino', linewidth = 2.0)
plt.plot(epochs, val_acc, 'r', label='Validação', linewidth = 2.0)
plt.title('Acurácia por época')
plt.xlabel('épocas')
plt.ylabel('porcentagem')
plt.grid(True)
plt.legend()
plt.show()

#GRÁFICO DE PERDA DE VALIDAÇÃO E TREINO
plt.plot(epochs, loss, 'b', label='Treino')
plt.plot(epochs, val_loss, 'r', label='Validação')
plt.title('Perda por época')
plt.xlabel('épocas')
plt.ylabel('porcentagem')
plt.grid(True)
plt.legend()
plt.show()

#CARREGA MODELO TREINADO

model=load_model('modelo.h5')

#MOSTRA COMO A REDE ESTÁ CONFIGURADA
sumario_modelo = model.summary()

#DIRETÓRIO DAS IMAGENS QUE DESEJA USAR COMO TESTE
imagens_teste = ['C:\\Users\\Brennus\\Downloads\\Dataset_Teste\\{}'.format(i) for i in os.listdir(diretorio_teste)]

#NÚMERO DE IMAGENS PRESENTE DENTRO DE imagens_teste
ImagensParaAvaliar = 40

#NÚMERO DE COLUNAS NA PLOTAGEM DAS IMAGENS DE TESTE
coluna_imagens_teste = 5

#PREDIÇÃO DO MODELO TREINADO COM AS IMAGENS DE TESTE
X_teste, y_teste = processar_imagens(imagens_teste[0:ImagensParaAvaliar]) #Y_test in this case will be empty.
x = np.array(X_teste)
teste_datagen = ImageDataGenerator(rescale=1. / 255)
i = 0
text_labels = []
plt.figure(figsize=(20,20))

for batch in teste_datagen.flow(x, batch_size=1):
    predicao = model.predict(batch)
    print(predicao)
    print(np.argmax(predicao))
    n = np.argmax(predicao)
    if n == 0:
        text_labels.append(f'Capacitor')
    elif n == 1:
        text_labels.append(f'Resistor')
    elif n == 2:
        text_labels.append(f'Transistor')
    elif n == 3:
        text_labels.append(f'IC')

    #PLOT DAS IMAGENS DE TREINO
    plt.subplot((ImagensParaAvaliar / coluna_imagens_teste) + 1, coluna_imagens_teste, i + 1)
    plt.title('' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % ImagensParaAvaliar == 0:
        break
plt.show()
