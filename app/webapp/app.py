#------------------------------------------BIBLIOTECAS--------------------------------------
from cgi import print_form
from enum import auto
import os
from tkinter import Button
from flask import Flask, render_template, request, url_for
from flask_dropzone import Dropzone

import urllib.request
import os
import zipfile
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from PIL import Image
#--------------------------------------------------------------------------------------------

#-------------------------------------VARIABLES GLOBALES-------------------------------------
basedir = os.path.abspath(os.path.dirname(__file__))

app=Flask(__name__)

app.config.update(
    UPLOADED_PATH= os.path.join(basedir,'uploads'))

dropzone = Dropzone(app)
app.config['DROPZONE_MAX_FILE_SIZE']=7

#Definicion global del modelo, para su entrenamiento
modelo_entrenar=None

#List que utilizo para asegurarme de que sólo se puede subir una imagen a la vez
cache=[]

#Contador de bajas
bajasTotales=0

#Rutas de las imagenes para en entrenamiento del modelo
img_dir = "Webapp_TFG\\Fotos_Proyecto_Detección_de_Bajas"
train_dir = os.path.join(img_dir, 'Entrenamiento')
validation_dir = os.path.join(img_dir, 'Validacion')
train_alive_dir = os.path.join(train_dir, 'Gallinas_vivas')
train_dead_dir = os.path.join(train_dir, 'Gallinas_muertas')
validation_alive_dir = os.path.join(validation_dir, 'Gallinas_vivas')
validation_dead_dir = os.path.join(validation_dir, 'Gallinas_muertas')

train_datagen = ImageDataGenerator(rotation_range=90, brightness_range=(0.2, 0.8), horizontal_flip=True, vertical_flip=True, rescale = 1.0/255.)
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=5,class_mode='binary',target_size=(150, 150))     
validation_generator =  test_datagen.flow_from_directory(validation_dir,batch_size=5,class_mode  = 'binary',target_size = (150, 150))

#Almacenar modelo
model_path = 'Webapp_TFG\\Modelo_guardado'
model_dir = os.path.dirname(model_path)
#--------------------------------------------------------------------------------------------

#-------------------------------------FUNCIÓN INICIAL, SUBIDA DE IMÁGENES--------------------
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(app.config['UPLOADED_PATH'],"imagen.jpg"))
        if len(cache)==0:
            cache.append(f)
        else:
            cache[0]=f

    return render_template('index.html')
#--------------------------------------------------------------------------------------------

#------------------------------------------CLASIFICAR IMÁGENES-------------------------------
@app.route('/clasificador',methods=['POST','GET'])
def clasificador():
    if request.method == 'POST':
        if request.form['boton_clasificar'] == 'Clasificar imagenes':
            mensaje=predicciones()
            mensaje_bajas=gestionContador(mensaje)
            return render_template('index.html',mensaje_predict=mensaje, mensaje_conta=mensaje_bajas)
#--------------------------------------------------------------------------------------------

#------------------------------------------REINICIAR CONTADOR--------------------------------
@app.route('/contador',methods=['POST','GET'])
def reiniciarContador():
    global bajasTotales
    if request.method == 'POST':
        if request.form['boton_contador'] == 'Resetear contador':
            bajasTotales=0
    mensaje_bajas=str(bajasTotales)
    return render_template('index.html', mensaje_conta=mensaje_bajas)
#--------------------------------------------------------------------------------------------

#------------------------------------------CAMBIAR VENTANA-ENTRENAR--------------------------
@app.route('/CambiarVentana1',methods=['POST','GET'])
def cambiarVentanaEntrenar():
    if request.method == 'POST':
        if request.form['boton_cambiar'] == 'Cambiar ventana':
            return render_template('entrenamiento.html', mensaje_estado="Seleccione una acción.")
#--------------------------------------------------------------------------------------------

#------------------------------------------CAMBIAR VENTANA-CLASIFICAR------------------------
@app.route('/CambiarVentana2',methods=['POST','GET'])
def cambiarVentanaClasificar():
    if request.method == 'POST':
        if request.form['boton_cambiar'] == 'Cambiar ventana':
            return render_template('index.html')
#--------------------------------------------------------------------------------------------

#------------------------------------------REENTRENAR EL MODELO------------------------------
@app.route('/entrenar_guardar',methods=['POST','GET'])
def entrenar_guardar():
    if request.method == 'POST':
        global modelo_entrenar
        if request.form['boton_entrenar_guardar'] == 'Entrenar modelo':
            modelo_entrenar=modelo()
            modelo_entrenar.compile(optimizer=RMSprop(learning_rate=0.001),loss='binary_crossentropy',metrics = ['accuracy'])
            history = modelo_entrenar.fit(train_generator,validation_data=validation_generator,epochs=2,verbose=2)
            resultado = modelo_entrenar.evaluate(validation_generator, verbose=0)
            precision_mensaje=resultado[1]
            return render_template('entrenamiento.html', mensaje_estado=f'Entrenamiento completo. Precisión: {precision_mensaje}')
        elif request.form['boton_entrenar_guardar'] == 'Guardar modelo':
            if modelo_entrenar is not None:
                modelo_entrenar.save(model_path)
                return render_template('entrenamiento.html', mensaje_estado="El modelo se ha guardado con éxito.")
            else:
                return render_template('entrenamiento.html', mensaje_estado="Por favor, entrene un modelo primero.")
        elif request.form['boton_entrenar_guardar'] == 'Comprobar precision':
            #Compruebo si ya existe un modelo. Si existe, devuelvo su precisión. Si no, compruebo si se ha entrenado uno.
            try:
                modelo_prueba=tf.keras.models.load_model(model_path)
                resultado = modelo_prueba.evaluate(validation_generator, verbose=0)
                precision_mensaje=resultado[1]
                return render_template('entrenamiento.html', mensaje_estado=f'Precisión: {precision_mensaje}.')
            except:
                modelo_prueba=modelo_entrenar
                #Si tampoco se ha entrenado un modelo, devuelvo un mensaje indicando que se tiene que entrenar uno.
                if modelo_prueba is not None:
                    modelo_prueba.compile(optimizer=RMSprop(learning_rate=0.001),loss='binary_crossentropy',metrics = ['accuracy'])
                    resultado = modelo_prueba.evaluate(validation_generator, verbose=0)
                    precision_mensaje=resultado[1]
                    return render_template('entrenamiento.html', mensaje_estado=f'Precisión: {precision_mensaje}.')
                else:
                    return render_template('entrenamiento.html', mensaje_estado='No hay ningún modelo entrenado.')
#--------------------------------------------------------------------------------------------

#--------------------------------------INCREMENTAR CONTADOR----------------------------------
def gestionContador(cadena):
    global bajasTotales
    if cadena == "Hay algún ave muerta en esta jaula":
        bajasTotales+=1
    return(str(bajasTotales))
#--------------------------------------------------------------------------------------------

#----------------------------------------DEFINICIÓN DEL MODELO-------------------------------
def modelo():
    modelo_def = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(80, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2), 
        
        tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(100, (3,3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(), 
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])

    return modelo_def
#--------------------------------------------------------------------------------------------

#-------------------------------------CLASIFICAR IMAGEN--------------------------------------
def predicciones():
    #Importar el modelo
    model=tf.keras.models.load_model(model_path)

    path="Webapp_TFG\\uploads\\imagen.jpg"
    img=image.image_utils.load_img(path, target_size=(150, 150))

    x=image.image_utils.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
  
    classes = model.predict(images)
    #print(f'RESULTADOS: {classes[0]}')
  
    if classes[0]>1.0e-17:
        mensajeFinal="No hay aves muertas en esta jaula"
    else:
        mensajeFinal="Hay algún ave muerta en esta jaula"

    return mensajeFinal
#--------------------------------------------------------------------------------------------

#-------------------------------------------DEBUG--------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
#--------------------------------------------------------------------------------------------