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
    
        img = cv2.imread(rutaImagen, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (120, 160))
        
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
    
       
    
####################################################################################

# RUTAS DE LA APLICACION WEB



@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        # Handle POST Request here
        return render_template('index.html')
    return render_template('index.html')



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



if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)   