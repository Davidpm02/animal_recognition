from flask import Flask, redirect, url_for, render_template, request, jsonify
import os
import cv2
import numpy as np
from PIL import Image, ImageFile
from io import BytesIO

import tensorflow as tf

ImageFile.LOAD_TRUNCATED_IMAGES = True

app=Flask(__name__)

UPLOAD_FOLDER = 'imagenesUsuario/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

####################################################################################

# FUNCIONES AUXILIARES

def cargarModelo():
    
    """Summary:Funcion que se encarga de cargar el modelo entrenado a utilizar a partir de los archivos
               .H5 y .JSON que generamos en el notebook de Jupyter.
       
       Args:
       
       Returns:
         - La funcion, tras cargar el modelo entrenado, lo retorna al scope global del programa.
       
       """
       
       
    
    ruta_H5 = 'modeloEntrenado/modeloCNN_entrenado.h5'
    ruta_JSON = 'modeloEntrenado/modeloCNN_entrenado.json'
    
    with open(ruta_JSON, 'r') as json_file:
        modelo_json = json_file.read()
    
    modeloCNN = tf.keras.models.model_from_json(modelo_json)
    
    # Cargo los pesos del modelo entrenado (almacenados en el archivo .H5)
    
    modeloCNN.load_weights(ruta_H5)
    
    return modeloCNN


# ----------------------------------------------------------------------------------------



def cargarModelo_perros():
    
    """Summary:Funcion que se encarga de cargar el modelo entrenado a utilizar a partir de los archivos
               .H5 y .JSON que generamos en el notebook de Jupyter.
       
       Args:
       
       Returns:
         - La funcion, tras cargar el modelo entrenado, lo retorna al scope global del programa.
       
       """
       
       
    
    ruta_H5 = 'modeloEntrenado_perros/modeloCNN_entrenado_PERROS.h5'
    ruta_JSON = 'modeloEntrenado_perros/modeloCNN_entrenado_PERROS.json'
    
    with open(ruta_JSON, 'r') as json_file:
        modelo_json = json_file.read()
    
    modeloCNN = tf.keras.models.model_from_json(modelo_json)
    
    # Cargo los pesos del modelo entrenado (almacenados en el archivo .H5)
    
    modeloCNN.load_weights(ruta_H5)
    
    return modeloCNN


# ----------------------------------------------------------------------------------------


def cargarModelo_mariposas():
    
    """Summary:Funcion que se encarga de cargar el modelo entrenado a utilizar a partir de los archivos
               .H5 y .JSON que generamos en el notebook de Jupyter.
       
       Args:
       
       Returns:
         - La funcion, tras cargar el modelo entrenado, lo retorna al scope global del programa.
       
       """
       
       
    
    ruta_H5 = 'modeloEntrenado_mariposas/modeloCNN_entrenado_MARIPOSAS.h5'
    ruta_JSON = 'modeloEntrenado_mariposas/modeloCNN_entrenado_MARIPOSAS.json'
    
    with open(ruta_JSON, 'r') as json_file:
        modelo_json = json_file.read()
    
    modeloCNN = tf.keras.models.model_from_json(modelo_json)
    
    # Cargo los pesos del modelo entrenado (almacenados en el archivo .H5)
    
    modeloCNN.load_weights(ruta_H5)
    
    return modeloCNN


# ----------------------------------------------------------------------------------------


def cargarModelo_aves():
    
    """Summary:Funcion que se encarga de cargar el modelo entrenado a utilizar a partir de los archivos
               .H5 y .JSON que generamos en el notebook de Jupyter.
       
       Args:
       
       Returns:
         - La funcion, tras cargar el modelo entrenado, lo retorna al scope global del programa.
       
       """
       
       

    ruta_H5 = 'modeloEntrenado_aves/modeloCNN_entrenado_AVES.h5'
    ruta_JSON = 'modeloEntrenado_aves/modeloCNN_entrenado_AVES.json'
    
    with open(ruta_JSON, 'r') as json_file:
        modelo_json = json_file.read()
    
    modeloCNN = tf.keras.models.model_from_json(modelo_json)
    
    # Cargo los pesos del modelo entrenado (almacenados en el archivo .H5)
    
    modeloCNN.load_weights(ruta_H5)
    
    return modeloCNN


# ----------------------------------------------------------------------------------------


def obtenerPrediccionModelo(modeloCNN, rutaImagen):
    
    """Funcion parametrizada que se encarga de convertir una imagen a escala de grises en array de numpy, normalizar los valores de cada pixel y lanzarla contra el modelo,
       obteniendo una prediccion de este.
       
       Args:
         - modeloCNN ==> modelo cargado desde los archivos .JSON y .H5 del directorio 'modeloEntrenado/'. 
         - rutaImagen ==> ruta generada tras alojar la imagen que se recibe del formulario.
         
       Returns:
         - clase_predicha ==> salida del modelo (valor numerico entre 0 y 9) que corresponde a la prediccion que el modelo ha generado para la imagen en cuestion.
                              Este valor numerico corresponde a una etiqueta en cuestion, que se procesara mas adelante.
    """
    
    try:
    
        img = cv2.imread(rutaImagen)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Expandir las dimensiones de la imagen para que coincidan con las expectativas del modelo
        img = np.expand_dims(img, axis=-1)

        img = img / 255
        
        predictions = modeloCNN.predict(np.expand_dims(img, axis=0))

        # Obtiene la clase predicha (la clase con la probabilidad más alta)
        clase_predicha = np.argmax(predictions, axis=1)

        # Obtiene la probabilidad de la clase predicha
        #probabilidad_predicha = predictions[0][clase_predicha]
        
        return clase_predicha

    except Exception as e:
        
        print('Ha ocurrido un error al alimentar al modelo con la imagen proporcionada ==>', e)


# -----------------------------------------------------------------------------------------

def generar_hashmapClases():
    
    """Funcion sin parametros que de generar un diccionario que parsea las posibles salidas que ofrece el modelo, y le asigna como valor la etiqueta correspondiente con la]
       que fue entrenado.
       
       Args:
       
       
       Returns:
         - dictClases ==> la funcion retorna el diccionario generado al scope global del programa.  Una vez alli, se podra utilizar para obtener el nombre de la especie correspondiente
                          a la salida que ha ofrecido el modelo para la imagen subida por el usuario al formulario HTML.
    """
    
    
    try:
    
        claves = list(range(0, 10))  # Creo una lista con numeros del 0 al 9, que corresponden a las salidas que ofrece el modelo entrenado.
        nombreEtiquetas = ['mariposa', 'gato', 'gallina', 'vaca', 'perro', 'elefante', 'caballo', 'arania', 'oveja', 'ardilla']
        dictClases = {}
        
        for indice, clave in enumerate(claves):
            dictClases[clave] = nombreEtiquetas[indice]
            
        
        print('Diccionario de mapeo de clases generado con exito.')
        
        return dictClases
    
    except Exception as e:
        
        print('Ha ocurrido un error al generar el hashmap que mapea las etiquetas del modelo ==>', e)



# -----------------------------------------------------------------------------------------


def generar_hashmapClases_perros():
    
    """Funcion sin parametros que de generar un diccionario que parsea las posibles salidas que ofrece el modelo, y le asigna como valor la etiqueta correspondiente con la]
       que fue entrenado.
       
       Args:
       
       
       Returns:
         - dictClases ==> la funcion retorna el diccionario generado al scope global del programa.  Una vez alli, se podra utilizar para obtener el nombre de la especie correspondiente
                          a la salida que ha ofrecido el modelo para la imagen subida por el usuario al formulario HTML.
    """
    
    
    try:
    
        claves = list(range(0, 10))  # Creo una lista con numeros del 0 al 9, que corresponden a las salidas que ofrece el modelo entrenado.
        nombreEtiquetas = ['american staffordshire', 'beagle', 'border collie', 'chihuahua', 'doberman', 'bulldog frances', 'labrador', 'rottweiler', 'toy terrier', 'yorkshire terrier']
        dictClases = {}
        
        for indice, clave in enumerate(claves):
            dictClases[clave] = nombreEtiquetas[indice]
            
        
        print('Diccionario de mapeo de clases generado con exito.')
        
        return dictClases
    
    except Exception as e:
        
        print('Ha ocurrido un error al generar el hashmap que mapea las etiquetas del modelo ==>', e)



# -----------------------------------------------------------------------------------------


def generar_hashmapClases_mariposas():
    
    """Funcion sin parametros que de generar un diccionario que parsea las posibles salidas que ofrece el modelo, y le asigna como valor la etiqueta correspondiente con la]
       que fue entrenado.
       
       Args:
       
       
       Returns:
         - dictClases ==> la funcion retorna el diccionario generado al scope global del programa.  Una vez alli, se podra utilizar para obtener el nombre de la especie correspondiente
                          a la salida que ha ofrecido el modelo para la imagen subida por el usuario al formulario HTML.
    """
    
    
    try:
    
        claves = list(range(0, 10))  # Creo una lista con numeros del 0 al 9, que corresponden a las salidas que ofrece el modelo entrenado.
        nombreEtiquetas = ['american snoot', 'an 88', 'common banded awl', 'mestra', 'monarch', 'orange oakleaf', 'paper kite', 'purple hairstreak', 'sleepy orange', 'southern dogface']
        dictClases = {}
        
        for indice, clave in enumerate(claves):
            dictClases[clave] = nombreEtiquetas[indice]
            
        
        print('Diccionario de mapeo de clases generado con exito.')
        
        return dictClases
    
    except Exception as e:
        
        print('Ha ocurrido un error al generar el hashmap que mapea las etiquetas del modelo ==>', e)



# -----------------------------------------------------------------------------------------


def generar_hashmapClases_aves():
    
    """Funcion sin parametros que de generar un diccionario que parsea las posibles salidas que ofrece el modelo, y le asigna como valor la etiqueta correspondiente con la]
       que fue entrenado.
       
       Args:
       
       
       Returns:
         - dictClases ==> la funcion retorna el diccionario generado al scope global del programa.  Una vez alli, se podra utilizar para obtener el nombre de la especie correspondiente
                          a la salida que ha ofrecido el modelo para la imagen subida por el usuario al formulario HTML.
    """
    
    
    try:
    
        claves = list(range(0, 10))  # Creo una lista con numeros del 0 al 9, que corresponden a las salidas que ofrece el modelo entrenado.
        nombreEtiquetas = ['mascarita de altamira', 'zapornia bicolor', 'malkoha azul', 'alcuela crestada', 'evening grosbeak', 'horned sungem', 'halcon peregrino', 'agapornis roseicollis', 'buho nevado', 'calao crestiblanco']
        dictClases = {}
        
        for indice, clave in enumerate(claves):
            dictClases[clave] = nombreEtiquetas[indice]
            
        
        print('Diccionario de mapeo de clases generado con exito.')
        
        return dictClases
    
    except Exception as e:
        
        print('Ha ocurrido un error al generar el hashmap que mapea las etiquetas del modelo ==>', e)



# -----------------------------------------------------------------------------------------







def guardar_imagen(bytes_imagen, ruta_archivo):
    try:
        imagen = Image.open(BytesIO(bytes_imagen))
        imagen.save(ruta_archivo)
        return True
    except Exception as e:
        print(f'Error al guardar la imagen: {str(e)}')
        return False

    
    
 # -----------------------------------------------------------------------------------------

def vaciarDirectorioImagenes():
    
    """Funcion sin parametros que elimina todo el contenido dentro del directorio definido.
    
       Args:
       
       
       Returns:
       
    """
    
    directorioImagenes = 'imagenesUsuario/'
    
    for archivo in os.listdir(directorioImagenes):
        os.remove(os.path.join(directorioImagenes, archivo))
        print('El archivo "{}" ha sido eliminado con exito.'.format(archivo))
        
        
        
# -----------------------------------------------------------------------------------------       

def leerArchivo_txt__general(etiquetaPredicha):
    
    directorioTextos = "textos_animales/general/"
    
    rutaCompleta = os.path.join(directorioTextos, etiquetaPredicha + '.txt')
    
    
    try:
    
        with open(rutaCompleta, 'r', encoding='utf-8') as archivo:
            contenidoArchivo = archivo.read()  # Variable que contiene el texto para la etiqueta predicha por el modelo
    
        return contenidoArchivo
    
    except Exception as e:
        
        print("Ha ocurrido un error al momento de leer el archivo .txt para la etiqueta {} ==> ".format(etiquetaPredicha), e)
        
    
####################################################################################

# RUTAS DE LA APLICACION WEB



@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        # Handle POST Request here
        return render_template('index.html')
    return render_template('index.html')


# -----------------------------------------------------------------------------------------


@app.route('/clasificador_perros',methods=['GET','POST'])
def index_clasificadorPerros():
    if request.method=='POST':
        # Handle POST Request here
        return render_template('clasificador_perros.html')
    return render_template('clasificador_perros.html')

# -----------------------------------------------------------------------------------------



@app.route('/clasificador_mariposas',methods=['GET','POST'])
def index_clasificadorMariposas():
    if request.method=='POST':
        # Handle POST Request here
        return render_template('clasificador_mariposas.html')
    return render_template('clasificador_mariposas.html')


# -----------------------------------------------------------------------------------------


@app.route('/clasificador_aves',methods=['GET','POST'])
def index_clasificadorAves():
    if request.method=='POST':
        # Handle POST Request here
        return render_template('clasificador_aves.html')
    return render_template('clasificador_aves.html')




# -----------------------------------------------------------------------------------------



@app.route('/clasificarImagen', methods=['GET', 'POST'])
def clasificarImagen():
    carpetaDestino = 'imagenesUsuario/'

    if not os.path.exists(carpetaDestino):
        os.mkdir(carpetaDestino)
    else:
        print('El directorio objetivo ya existe.')

    print('Metodo de solicitud ==>', request.method)

    if request.method == 'POST':
        archivo = request.files['imagen']

        # Verifica si se ha subido un archivo
        if archivo.filename != '':
            
            nombre_archivo = 'imagen_subida.png'  # Nombre del archivo de salida
            ruta_archivo = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)

            # Guarda el archivo en la carpeta como una imagen
            if guardar_imagen(archivo.read(), ruta_archivo):
                
                modeloCNN = cargarModelo()
                dictClases = generar_hashmapClases()
                
                
                etiqueta_predicha =  obtenerPrediccionModelo(modeloCNN, ruta_archivo)  # array con valor numerico ==> salida del modelo
                print('Etiqueta predicha ==> ', etiqueta_predicha)
                clase_predicha = dictClases[etiqueta_predicha[0]]  # string ==> nombre etiqueta predicha (especie animal)
                print(clase_predicha)
                
                texto_clasePredicha = leerArchivo_txt__general(clase_predicha)
                print(texto_clasePredicha)
                
                # .
                # .
                # .
                
                # Supongamos que etiqueta_predicha y texto_clase_predicha son tus datos
                response_data = {'etiquetaPredicha': clase_predicha, 'texto_clasePredicha': texto_clasePredicha}

                # Devuelve la respuesta como un objeto JSON
                return jsonify(response_data)
            

            else:
                return 'Error al procesar la imagen.'
        else:

            return 'No se ha subido ningún archivo.'

    #return redirect('index.html')
    
    
    if len(os.listdir('imagenesUsuario/')) > 0:
        vaciarDirectorioImagenes()
        
    else:
        print('El directorio objetivo ya esta vacio.')
        
        
        

# -----------------------------------------------------------------------------------------



@app.route('/clasificarImagen_perros', methods=['GET', 'POST'])
def clasificarImagen_perros():
    carpetaDestino = 'imagenesUsuario/'

    if not os.path.exists(carpetaDestino):
        os.mkdir(carpetaDestino)
    else:
        print('El directorio objetivo ya existe.')

    print('Metodo de solicitud ==>', request.method)

    if request.method == 'POST':
        archivo = request.files['imagen']

        # Verifica si se ha subido un archivo
        if archivo.filename != '':
            
            nombre_archivo = 'imagen_subida.png'  # Nombre del archivo de salida
            ruta_archivo = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)

            # Guarda el archivo en la carpeta como una imagen
            if guardar_imagen(archivo.read(), ruta_archivo):
                
                modeloCNN = cargarModelo_perros()
                dictClases = generar_hashmapClases_perros()
                
                
                etiqueta_predicha =  obtenerPrediccionModelo(modeloCNN, ruta_archivo)  # array con valor numerico ==> salida del modelo
                print('Etiqueta predicha ==> ', etiqueta_predicha)
                clase_predicha = dictClases[etiqueta_predicha[0]]  # string ==> nombre etiqueta predicha (especie animal)
                print(clase_predicha)
                
                # .
                # .
                # .

                # Retorna una respuesta (por ejemplo, la ruta del archivo guardado)
                return clase_predicha
            
                
            else:
                return 'Error al procesar la imagen.'
        else:

            return 'No se ha subido ningún archivo.'

    #return redirect('index.html')
    
    
    if len(os.listdir('imagenesUsuario/')) > 0:
        vaciarDirectorioImagenes()
        
    else:
        print('El directorio objetivo ya esta vacio.')
        
        
             
        
 # -----------------------------------------------------------------------------------------
       
        

@app.route('/clasificarImagen_mariposas', methods=['GET', 'POST'])
def clasificarImagen_mariposas():
    carpetaDestino = 'imagenesUsuario/'

    if not os.path.exists(carpetaDestino):
        os.mkdir(carpetaDestino)
    else:
        print('El directorio objetivo ya existe.')

    print('Metodo de solicitud ==>', request.method)

    if request.method == 'POST':
        archivo = request.files['imagen']

        # Verifica si se ha subido un archivo
        if archivo.filename != '':
            
            nombre_archivo = 'imagen_subida.png'  # Nombre del archivo de salida
            ruta_archivo = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)

            # Guarda el archivo en la carpeta como una imagen
            if guardar_imagen(archivo.read(), ruta_archivo):
                
                modeloCNN = cargarModelo_mariposas()
                dictClases = generar_hashmapClases_mariposas()
                
                
                etiqueta_predicha =  obtenerPrediccionModelo(modeloCNN, ruta_archivo)  # array con valor numerico ==> salida del modelo
                print('Etiqueta predicha ==> ', etiqueta_predicha)
                clase_predicha = dictClases[etiqueta_predicha[0]]  # string ==> nombre etiqueta predicha (especie animal)
                print(clase_predicha)
                
                # .
                # .
                # .

                # Retorna una respuesta (por ejemplo, la ruta del archivo guardado)
                return clase_predicha
            else:
                return 'Error al procesar la imagen.'
        else:

            return 'No se ha subido ningún archivo.'

    #return redirect('index.html')
    
    
    if len(os.listdir('imagenesUsuario/')) > 0:
        vaciarDirectorioImagenes()
        
    else:
        print('El directorio objetivo ya esta vacio.')
        
                
        
# -----------------------------------------------------------------------------------------
      
        

@app.route('/clasificarImagen_aves', methods=['GET', 'POST'])
def clasificarImagen_aves():
    carpetaDestino = 'imagenesUsuario/'

    if not os.path.exists(carpetaDestino):
        os.mkdir(carpetaDestino)
    else:
        print('El directorio objetivo ya existe.')

    print('Metodo de solicitud ==>', request.method)

    if request.method == 'POST':
        archivo = request.files['imagen']

        # Verifica si se ha subido un archivo
        if archivo.filename != '':
            
            nombre_archivo = 'imagen_subida.png'  # Nombre del archivo de salida
            ruta_archivo = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)

            # Guarda el archivo en la carpeta como una imagen
            if guardar_imagen(archivo.read(), ruta_archivo):
                
                modeloCNN = cargarModelo_aves()
                dictClases = generar_hashmapClases_aves()
                
                
                etiqueta_predicha =  obtenerPrediccionModelo(modeloCNN, ruta_archivo)  # array con valor numerico ==> salida del modelo
                print('Etiqueta predicha ==> ', etiqueta_predicha)
                clase_predicha = dictClases[etiqueta_predicha[0]]  # string ==> nombre etiqueta predicha (especie animal)
                print(clase_predicha)
                
                # .
                # .
                # .

                # Retorna una respuesta (por ejemplo, la ruta del archivo guardado)
                return clase_predicha
            else:
                return 'Error al procesar la imagen.'
        else:

            return 'No se ha subido ningún archivo.'

    #return redirect('index.html')
    
    
    if len(os.listdir('imagenesUsuario/')) > 0:
        vaciarDirectorioImagenes()
        
    else:
        print('El directorio objetivo ya esta vacio.')



###########################################################################



if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)   