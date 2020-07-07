#------- Importação de bibliotecas ---------
import os
import gc 
import random
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from IPython import get_ipython
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
#----------------------------------------

#------ Inicialização de variáveis de configuração -----------
# Quantidade de épocas
Epocas = 50

# Quantidade de imagens de treino
NumeroIndutor = 3918
NumeroLed = 4253
NumeroCapacitor = 4729
NumeroResistor = 4729
NumeroTransistor = 5155

# Diretório de Treino e Teste 
train_dir = 'C:\\Users\\Lili\\Documents\\GitHub\\cnn-pibic\\imagens\\treino2'
#'C:\\Users\\Brennus\\Downloads\\treino_cnn_CRT'
test_dir = 'C:\\Users\\Lili\\Documents\\GitHub\\cnn-pibic\\imagens\\teste'
#'C:\\Users\\Brennus\\Desktop\\leds'

# Numero de colunas a serem exibidas as imagens
columns = 5

# Dimensões da imagem
nrows = 150
ncolumns = 150

#---------------------------------------------------------------
#função para deletar variáveis e chamar o garbage collector para salvar memória
def deallocate (arr_name_variable = []):
    for variable in arr_name_variable:
        variable = "del "+ variable
        exec(variable, globals(), globals())
    gc.collect()

# Lê e processa a imagem para o formato padrão para o treino 
def read_and_process_image(list_of_images):
    resizedImg = []  # array de imagens redimensionadas
    labelImg = []  # labels da imagem

    for image in list_of_images:
        try:
            # Redimensiona as Imagens de treino para 150x150 com 3 canais de cores (RGB)
            # Método de interpolação para realizar o zoom da imagem
            resizedImg.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR),(nrows, ncolumns), interpolation=cv2.INTER_CUBIC))

            #Enumera as labels
            if 'Capacitor' in image:
                labelImg.append(0)
            elif 'Resistor' in image:
                labelImg.append(1)
            elif 'Transistor' in image:
                labelImg.append(2)
            elif 'led' in image:
                labelImg.append(3)
            elif 'Indutor' in image:
                labelImg.append(4)

        # Caso ele não consiga ler as imagens (cv2.imread) elas vão ser ignoradas.
        except Exception as e:
            print(str(e))

    return resizedImg, labelImg
#------------------------------------------------------------------

#Cria um novo modelo e treina a rede
def treinaRede():
    #---------------- MODELO DE TREINO ----------------------------
    # Coleta imagens de treino com nome de Capacitor, Resistor e Transistor
    treino_capacitor = []
    treino_resistor = []
    treino_transistor = []
    treino_led = []
    treino_indutor = []

    for i in os.listdir(train_dir) :
        treino = train_dir+'\\{}'.format(i)

        if('Capacitor' in i):
            treino_capacitor.append(treino)
        elif('Resistor' in i):
            treino_resistor.append(treino)
        elif('Transistor' in i):
            treino_transistor.append(treino)
        elif('led' in i):
            treino_led.append(treino)
        elif('Indutor' in i):
            treino_indutor.append(treino)

    # Coleta as imagens que vão ser utilizadas para teste
    test_imgs = [test_dir+'\\{}'.format(i) for i in os.listdir(test_dir)]  

    # Monta o array com as imagens de treino carregadas posteriormente
    train_imgs = treino_capacitor[:NumeroCapacitor] + treino_resistor[:NumeroResistor] + treino_transistor[:NumeroTransistor]

    # Desorganiza as imagens
    random.shuffle(train_imgs)  

    # deallocate(["treino_capacitor", "treino_resistor", "treino_transistor", "treino_led", "treino_indutor"])

    del treino_capacitor
    del treino_resistor
    del treino_transistor

    # Recebe a lista de imagens e suas respectivas labels
    resizedImg, labelImg = read_and_process_image(train_imgs)

    # Converte os resultados de lista em um numpy array
    resizedImg = np.array(resizedImg)
    labelImg = np.array(labelImg)

    # deallocate(["train_imgs"])
    del train_imgs
    gc.collect()

    print("Formato(shape) das imagens de treino:", resizedImg.shape)
    print("Formato(shape) dos rótulos          :", labelImg.shape)

    # train_test_split realiza a divisão do dataset de treino entre: dataset de treino e dataset de validação.
    X_train, X_val, y_train, y_val = train_test_split(resizedImg, labelImg, test_size=0.20, random_state=2)

    print("Shape of train images is:", X_train.shape)
    print("Shape of validation images is:", X_val.shape)
    print("Shape of labels is:", y_train.shape)
    print("Shape of labels is:", y_val.shape)

    # deallocate(["resizedImg","labelImg"])
    del resizedImg
    del labelImg
    gc.collect()

    # get the length of the train and validation data
    ntrain = len(X_train)
    nval = len(X_val)

    # We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
    batch_size = 32


    # Criar modelo
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    # Diluição para regular a rede neural
    model.add(layers.Dropout(0.5))  
    model.add(layers.Dense(512, activation='relu'))

    # Mudar o numero de neuronios da última camada e colocar a função softmax de ativação
    model.add(layers.Dense(5, activation='softmax'))

    # Mostra o modelo
    model.summary()

    # We'll use the RMSprop optimizer with a learning rate of 0.0001
    # We'll use binary_crossentropy loss because its a binary classification
    # optimizers.RMSprop(lr=1e-4)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


    # Configuração de data augmentation a fim de aumentar a generalidade do modelo
    # Evita o sobreajuste, permite a previsão dos novos resultados
    train_datagen = ImageDataGenerator(rescale=1./255,  # Reescala a imagem entre 0 e 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

    # Há somente a reescala e não há a augmentação da validação
    val_datagen = ImageDataGenerator(rescale=1./255)


    # Gera os treinos
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    #---------------- FIM MODELO ----------------------------

    
    #--------- TREINAMENTO ---------------------
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

    # --------- FIM TREINAMENTO ---------------------
    

def main():
    opc = ''

    print("Deseja criar um novo modelo? Y/N")
    input(opc)

    if opc.lower() == 'y' :
        treinaRede()
        
    elif opc.lower() == 'n':
        #Carrega o modelo
        model = load_model('modelo_CRTLI.h5')

        # Diretório das imagens de teste
        # Ao mudar o diretório de teste não esqueça de mudar as variáveis test_dir e ImagensParaAvaliar.
        test_imgs = [test_dir+'\\{}'.format(i) for i in os.listdir(test_dir)]

        TirarFoto = False
        ImagensParaAvaliar = 11

        if (TirarFoto == False):
            # Now lets predict on the first ImagensParaAvaliar of the test set
            # Y_test in this case will be empty.
            X_test, y_test = read_and_process_image(test_imgs[0:ImagensParaAvaliar])
            resizedImg = np.array(X_test)
            test_datagen = ImageDataGenerator(rescale=1./255)
            i = 0
            text_labels = []
            plt.figure(figsize=(20, 20))
            
            for batch in test_datagen.flow(resizedImg, batch_size=1):
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
            # Não realizei teste nessa parte do código, então ele parmanece inalterado
            camera_port = 0
            file = 'C:\\Home\\usuario\\python\\Youtube CNN GatoCachorro Gera Modelo\\input\\test\\aaImagem.bmp'
            while(True):
                # tira foto da WebCam
                camera = cv2.VideoCapture(camera_port)
                retval, img = camera.read()
                cv2.imwrite(file, img)
                camera.release()
                test_imgs = [test_dir+'\\{}'.format(i) for i in os.listdir(test_dir)]
                X_test, y_test = read_and_process_image(test_imgs[0:ImagensParaAvaliar])
                resizedImg = np.array(X_test)
                test_datagen = ImageDataGenerator(rescale=1./255)
                i = 0
                text_labels = []
                plt.figure(figsize=(31, 31))
                for batch in test_datagen.flow(resizedImg, batch_size=1):
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





    sys.exit()


if __name__ == "__main__":
    main()
    pass
