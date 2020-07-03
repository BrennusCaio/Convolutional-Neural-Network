Epocas               = 50
NumeroIndutor = 3918
NumeroLed = 4253
NumeroCapacitor = 4729
NumeroResistor = 4729
NumeroTransistor = 5155

#Numero de colunas a serem exibidas as imagens
columns = 5

#import time para usar => #time.sleep(1)
import cv2
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
# Original: %matplotlib inline 
'exec(%matplotlib inline)'

#To see our directory
import os
import random
import gc   #Garbage collector for cleaning deleted data from memory

#Diretório de Treino e Teste
train_dir = 'C:\\Users\\Brennus\\Downloads\\treino_cnn_CRT'
test_dir = 'C:\\Users\\Brennus\\Desktop\\leds'

#Coleta imagens de treino com nome de Capacitor, Resistor e Transistor
treino_capacitor = ['C:\\Users\\Brennus\\Downloads\\treino_cnn_CRT\\{}'.format(i) for i in os.listdir(train_dir) if 'Capacitor' in i]
treino_resistor = ['C:\\Users\\Brennus\\Downloads\\treino_cnn_CRT\\{}'.format(i) for i in os.listdir(train_dir) if 'Resistor' in i]
treino_transistor = ['C:\\Users\\Brennus\\Downloads\\treino_cnn_CRT\\{}'.format(i) for i in os.listdir(train_dir) if 'Transistor' in i]
treino_led = ['C:\\Users\\Brennus\\Downloads\\treino_cnn_CRT\\{}'.format(i) for i in os.listdir(train_dir) if 'led' in i]
treino_indutor = ['C:\\Users\\Brennus\\Downloads\\treino_cnn_CRT\\{}'.format(i) for i in os.listdir(train_dir) if 'Indutor' in i]

#Coleta as imagens que vão ser utilizadas para teste
test_imgs = ['C:\\Users\\Brennus\\Downloads\\teste_sem_ruido_cnn\\{}'.format(i) for i in os.listdir(test_dir)] #get test images

# slice the dataset and use each class
train_imgs = treino_capacitor[:NumeroCapacitor] + treino_resistor[:NumeroResistor] + treino_transistor[:NumeroTransistor]
random.shuffle(train_imgs)  # shuffle it randomly

#Clear list that are useless
del treino_capacitor
del treino_resistor
del treino_transistor
gc.collect()   #collect garbage to save memory

#Lets declare our image dimensions
#we are using coloured images. 
nrows = 150
ncolumns = 150

#A function to read and process the images to an acceptable format for our model

def read_and_process_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = [] # labels


    for image in list_of_images:
        try:
            #Redimensiona as Imagens de treino para 150x150 com 3 canais de cores (RGB)
            #Método de interpolação para realizar o zoom da imagem
            X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
            #get the labels
            if 'Capacitor' in image:
                y.append(0)
            elif 'Resistor' in image:
                y.append(1)
            elif 'Transistor' in image:
                y.append(2)
            elif 'led' in image:
                y.append(3)
            elif 'Indutor' in image:
                y.append(4)

        #Caso ele não consiga ler as imagens (cv2.imread) elas vão ser ignoradas.
        except Exception as e:
            print(str(e))
    
    return X, y

#get the train and label data
X, y = read_and_process_image(train_imgs)

#Convert list to numpy array
X = np.array(X)
y = np.array(y)

del train_imgs
gc.collect()

#import seaborn as sns
#Plotagem do número de rótulos existentes no código
#sns.countplot(y)
#plt.title('Rótulos para capacitor e resistor:')

print("Formato(shape) das imagens de treino:", X.shape)
print("Formato(shape) dos rótulos          :", y.shape)

#train_test_split realiza a divisão do dataset de treino entre: dataset de treino e dataset de validação.
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)

#clear memory
del X
del y
gc.collect()

#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

#Criar modelo
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
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

#mudar o numero de neuronios da última camada e colocar a função softmax de ativação
model.add(layers.Dense(5, activation='softmax'))  #Sigmoid function at the end because we have just two classes

#Lets see our model
model.summary()

#We'll use the RMSprop optimizer with a learning rate of 0.0001
#We'll use binary_crossentropy loss because its a binary classification
#optimizers.RMSprop(lr=1e-4)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


#Lets create the augmentation configuration
#This helps prevent overfitting, since we are using a small dataset
train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale


#Create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


'''
#INICIO da Parte de Treinamento
#100 steps per epoch
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=Epocas,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)
#Salvar Modelo treinado
model.save('modelo_CRTLI.h5')

#lets plot the train and val curve
#get the details form the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Treino')
plt.plot(epochs, val_acc, 'r', label='Validação')
plt.title('Acurácia por época')
plt.xlabel('épocas')
plt.ylabel('porcentagem')
plt.legend()
plt.show()

#Train and validation loss
plt.plot(epochs, loss, 'b', label='Treino')
plt.plot(epochs, val_loss, 'r', label='Validação')
plt.title('Perda por época')
plt.xlabel('épocas')
plt.ylabel('porcentagem')
plt.legend()
plt.show()

# FIM da Parte de Treinamento

'''

#%%
from keras.models import load_model
model=load_model('modelo_CRTLI.h5')

#Diretório das imagens de teste
#Ao mudar o diretório de teste não esqueça de mudar as variáveis test_dir e ImagensParaAvaliar.
test_imgs = ['C:\\Users\\Brennus\\Desktop\\leds\\{}'.format(i) for i in os.listdir(test_dir)]

TirarFoto=False
ImagensParaAvaliar = 20

if (TirarFoto==False):
    #Now lets predict on the first ImagensParaAvaliar of the test set
    X_test, y_test = read_and_process_image(test_imgs[0:ImagensParaAvaliar]) #Y_test in this case will be empty.
    x = np.array(X_test)
    test_datagen = ImageDataGenerator(rescale=1./255)
    i = 0
    text_labels = []
    plt.figure(figsize=(20,20))

    for batch in test_datagen.flow(x, batch_size=1):
        pred = model.predict(batch)
        print(pred)
        print(np.argmax(pred))
        n = np.argmax(pred)
        if n == 0:
            text_labels.append(f'Capacitor')
        elif n == 1:
            text_labels.append(f'Resistor')
        elif n == 2:
            text_labels.append(f'Transistor')
        elif n == 3:
            text_labels.append(f'Led')
        elif n == 4:
            text_labels.append(f'Indutor')

        # Número de linhas, número de colunas
        plt.subplot((ImagensParaAvaliar / columns) + 1, columns, i + 1)
        plt.title('' + text_labels[i])
        imgplot = plt.imshow(batch[0])
        i += 1
        if i % ImagensParaAvaliar == 0:
            break
    plt.show()
else:
    #Não realizei teste nessa parte do código, então ele parmanece inalterado
    camera_port = 0
    file = 'C:\\Home\\usuario\\python\\Youtube CNN GatoCachorro Gera Modelo\\input\\test\\aaImagem.bmp'
    while(True):
        #tira foto da WebCam
        camera = cv2.VideoCapture(camera_port)
        retval, img = camera.read()
        cv2.imwrite(file,img)
        camera.release()
        test_imgs = ['C:\\Home\\usuario\\python\\Youtube CNN GatoCachorro Gera Modelo\\input\\test\\{}'.format(i) for i in os.listdir(test_dir)]
        X_test, y_test = read_and_process_image(test_imgs[0:ImagensParaAvaliar]) 
        x = np.array(X_test)
        test_datagen = ImageDataGenerator(rescale=1./255)
        i = 0
        text_labels = []
        plt.figure(figsize=(31,31))
        for batch in test_datagen.flow(x, batch_size=1):
            pred = model.predict(batch)
            if pred > 0.7:
                text_labels.append(f'Cachorro {pred}')
            elif pred < 0.3:
                text_labels.append(f'Gato {pred}')
            else:     
                text_labels.append('?')
            plt.subplot((ImagensParaAvaliar / columns) + 1, columns, i + 1)
            plt.title('' + text_labels[i])
            get_ipython().magic('clear')
            imgplot = plt.imshow(batch[0])
            i += 1
            if i % ImagensParaAvaliar == 0:
                break
        plt.show()
# Se necessario apagar a foto tirada pela Web Cam        
#        if(os.path.isfile(file)):
#            os.remove(file)
