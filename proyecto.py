import os
import matplotlib.pyplot as plt
from PIL import Image

#directorio donde esttán las imágenes
base_dir = "./imgs"

#creamos una lista que nos permitirá ver la cantidad de imágenes por cada categoría
num_images_per_category = []

#Obtenemos la cantidad de impagenes que hay en cada una de las categorías, las cuales están como folders y vamos iterando para ver la cantidad por folder
folders = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    num_images = len(os.listdir(folder_path))
    num_images_per_category.append(num_images)

#Etiquetas para las categorías, las cuales son las 4 que ya se mencionaron
labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

#gráfica de distribución para ver la cantidad
plt.figure(figsize=(8, 6))
plt.bar(labels, num_images_per_category, color='skyblue')
plt.xlabel('Categoría')
plt.ylabel('Número de imágenes')
plt.title('Distribución de imágenes por categoría')
plt.show()

#Por último, veremos de manera textual la cantidad de datos que se tienen por categoría para poder hacernos una idea si están balanceadas o desbalanceadas
#pequeño spoiler, no están nada balanceadas
folders = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    num_images = len(os.listdir(folder_path))
    print(f"Número de imágenes en '{folder}':", num_images)


#COMPROBAREMOS EL TAMAÑO DE LAS IMÁGENES


#Volvemos a cargar el directorio donde se encuentran las imágenes, está será la última vez que comentemos esto, de ahora en más cada que se use en el código
#ya no lo vamos a mencionar
base_dir = "./imgs/"

#Creamos una lista donde se va a estar almacenando el tamaño de las imágenes, es decir, largo y ancho
image_sizes = []


#vamos a hacer primero una iteración en cada folder, de nuevo
folders = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    #y ahora vamos a iterar en cada imagen de cada folder, obteniendo la dirección del folder y el nombre de la imágen, realmente el nombre no es muy útil
    #pero la función necesitaba dos parámetros y ps colocamos este para no tener problemas xd
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        #una vez que se tenga una imgaen, calculamos el largo y ancho y lo agregamos a la lista creada arriba
        with Image.open(image_path) as img:
            width, height = img.size
            image_sizes.append((width, height))

#Separar los tamaños de ancho y largo, la primera entrada es el ancho de la imagen y la segunda entrada el alto
widths = [size[0] for size in image_sizes]
heights = [size[1] for size in image_sizes]

#para poder ver si hay diferencias entre los tamaños, hacemos una gráfica que nos muestre todos los tamaños de todas las imágenes, pequeño spoiler de nuevo, todas son
#el mismo tamaño, es una muy buena base para empezar
plt.figure(figsize=(10, 6))
plt.scatter(widths, heights, color='skyblue', alpha=0.5)
plt.title('Tamaño de todas las imágenes')
plt.xlabel('Ancho (píxeles)')
plt.ylabel('Alto (píxeles)')
plt.grid(True)
plt.show()


#vamos a recortar todas las imágenes originales, y crear un directorio donde estarán las recortadas, ya que son las que nos interesa conservar para el
#análisis y evaluación del modelo

base_dir = "./imgs"

#En este directorio donde se guardarán las imágenes recortadas, que se llama como el de imgs, pero cropped que significa recortado en inglés
cropped_dir = "./imgs_cropped"
os.makedirs(cropped_dir, exist_ok=True)

#Vamos a seleccionar dos parámetros, crop top y bottom, es decir, recorte por abajo y recorte por arriba, ya que esta es la parte de la radiografía que sobra
crop_top = 16
crop_bottom = 16

#Nuevamente, vamos a iterar por cada folder y cada imagen dentro del folder, y una vez más como en el caso de cargar el directorio, ya no lo vamos a mencionar, porque
#pues este proceso ya se hizo varias veces y es cansado estar comentando a cada rato esto xd
folders = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        #Una vez que ya se abrió la imagen
        with Image.open(image_path) as img:
            #calculamos el tamaño original, el cual es 176 x 208
            width, height = img.size

            #Y ahora vamos a recortar la imagen, todas las imágenes
            img_cropped = img.crop((0, crop_top, width, height - crop_bottom))

            #Guardar la imagen recortada en el nuevo directorio, y posteriormente como estamos iterando, se van a guardar todas y cada una de las imágenes recortadas
            #en este mismo directorio
            cropped_image_path = os.path.join(cropped_dir, folder, image_name)
            os.makedirs(os.path.dirname(cropped_image_path), exist_ok=True)
            img_cropped.save(cropped_image_path)

#todos estos folders de imágenes se encuentran en la carpeta de archivos de google colab :D

"""Vamos a verificar que sí se nos recortó bien las imágenes:"""

#es el mismo código de la primera vez namas con el cambio de agarrar imágenes recortadas jaja

#Volvemos a cargar el directorio donde se encuentran las imágenes, está será la última vez que comentemos esto, de ahora en más cada que se use en el código
#ya no lo vamos a mencionar
cropped_dir = "./imgs_cropped"

#Creamos una lista donde se va a estar almacenando el tamaño de las imágenes, es decir, largo y ancho
cropped_image_sizes = []


#vamos a hacer primero una iteración en cada folder, de nuevo
folders = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
for folder in folders:
    folder_path = os.path.join(cropped_dir, folder)
    #y ahora vamos a iterar en cada imagen de cada folder, obteniendo la dirección del folder y el nombre de la imágen, realmente el nombre no es muy útil
    #pero la función necesitaba dos parámetros y ps colocamos este para no tener problemas xd
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        #una vez que se tenga una imgaen, calculamos el largo y ancho y lo agregamos a la lista creada arriba
        with Image.open(image_path) as img:
            width, height = img.size
            cropped_image_sizes.append((width, height))

#Separar los tamaños de ancho y largo, la primera entrada es el ancho de la imagen y la segunda entrada el alto
widths = [size[0] for size in cropped_image_sizes]
heights = [size[1] for size in cropped_image_sizes]

#para poder ver si hay diferencias entre los tamaños, hacemos una gráfica que nos muestre todos los tamaños de todas las imágenes, pequeño spoiler de nuevo, todas son
#el mismo tamaño, es una muy buena base para empezar
plt.figure(figsize=(10, 6))
plt.scatter(widths, heights, color='skyblue', alpha=0.5)
plt.title('Tamaño de todas las imágenes ya recortadas')
plt.xlabel('Ancho (píxeles)')
plt.ylabel('Alto (píxeles)')
plt.grid(True)
plt.show()

"""Y como podemos notar, efectivamente se tiene el mismo tamaño, por lo que hasta el momento nuestro preprocesamiento va bien.

### Patrón de colores en las imágenes

Necesitamos ver si todas las imágenes tienen una tonalidad parecida, puesto que esto nos va a permitir tener un proceso de clasificación más coherente y que pueda ser más confiable para este tipo de problemas sobre clasificar radiografías, ya que la confianza en el modelo debe ser lo mejor posible, aunque claro, al hacer esto vamos a perder variabilidad en los patrones de color si queremos ingresar otros datos, pero usualmente las radiografías siempre suelen tener el mismo tono, por lo que creemos esto será una buena idea e implementación.

Esto lo haremos usando el sistema de tonalidad de grises, ya que nuestras radiografías están en tonos de grises, donde se usa un solo canal de color para representar la intensidad, yendo ed 0 a 255, donde el 0 es negro absoluto y 255 blanco absoluto, además tener todo en esta escala ayuda a no cargar tanto costo computacional.
"""

#esto no lo vamos a comentar, es donde están las imágenes recortadas
cropped_dir = "./imgs_cropped"

#lista donde vamos a almacenar la tonalidad en escala de grises para cada imagen
average_brightness_values = []

#Iterar sobre las carpetas, de nuevo, no haremos mucho comentario
folders = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
for folder in folders:
    folder_path = os.path.join(cropped_dir, folder)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        #para cada una de las imágenes
        with Image.open(image_path) as img:
            #Convertir la imagen a escala de grises
            img_gray = img.convert('L')

            #Y ahora vamos a calcular la tonalidad promedio de los píxeles en escala de grises, para obtener el brillo
            brightness = img_gray.getdata()
            average_brightness = sum(brightness) / len(brightness)
            average_brightness_values.append(average_brightness)

#Grafica
plt.figure(figsize=(10, 6))
plt.hist(average_brightness_values, bins=50, color='skyblue', edgecolor='black')
plt.title('Tonalidad de las Imágenes Recortadas')
plt.xlabel('Tonalidad Promedio')
plt.ylabel('Número de Imágenes')
plt.grid(True)
plt.show()

"""Podemos ver que la mayoría de las imágenes tiene un tono de grises parecido, cerca del 80 en el valor de escala de grises, no vamos a hacer modificaciones para tener todas del mismo tono, puesto que estaríamos quitando mucha generalidad al modelo, dejándolo así podemos tener un poco más de variabilidad en los datos que nos puede ayudar a obtener mejores resultados.

## Elección del modelo

Ya tenemos los datos con su primera preprocesación, la general que es para poder implementarlo a cualquier tipo de modelo, pero bien, ahora la pregunta es, ¿Qué modelo usar?

Bueno, sabemos de primera mano que por la cantidad de datos y etiquetas lo mejor a usar podrían ser Redes Neuronales, usando deeplearning, pero este proceso es muy tardado y conlleva alto costo computacional, es por esto que veremos si podemos escoger otro modelo que sea más simple en implementar y ver si tiene buenos resultados.

Proponemos el SVM, este modelo es eficaz cuando se tienen grandes dimensiones de datos, en este caso tenemos imágenes, donde las dimensiones son los pixeles y tenemos 176x176, es por esto que hemos optado por usar este modelo, más aún haremos como tal dos modelos, primero uno donde probaremos el modelo sin realizar una disminución de dimensiones, y luego uno donde applicaremos PCA para ver si hay alguna mejora, para cada uno haremos ajustes de hiperparámetros necesarios que nos permitan obtener el mejor modelo.
"""

### Normalización de las imágenes ya procesadas


#Descargar la librería scikit-image para poder normalizar los pixeles
from skimage import io, transform
import numpy as np

#directorio
cropped_dir = "/content/ia-2024/imgs_cropped"

#Otra lista más, ahora para almacenar las imágenes recortadas
normalized_images = []

#iteracion en todas las imagenes en todos los folders
folders = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
for folder in folders:
    folder_path = os.path.join(cropped_dir, folder)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        #vamos a cargar todas las imágenes que vamos teniendo
        img = io.imread(image_path)

        #y normalizamos los pixeles de cada una de ellas, restando los pixeles originales, menos la media entre la desviación estándar de los pixeles en la imagen
        img_normalized = (img - np.mean(img)) / np.std(img)

        #vamos añadiendo cada imagen normalizada a nuestra lista
        normalized_images.append(img_normalized)

### Aplanamiento de Imágenes:

#Nuestras imágenes, al ser imágenes, tienen muchas dimensiones, debemos de realizar un aplanado que nos permitirá tener las imágenes como conjuntos de características bidimensionales, que es la entrada que espera nuestro modelo de SVM, para solucionar esto tenemos que hacer un aplanamiento de imágenes, para que puedan ser tratadas como vectores de características.

#Hacemos una lista donde se irán almacenando las imágenes aplanadas
flattened_images = []

#hacemos otra itreacion, pero en este caso iteramos sobre todas las imágenes que ya fueron normalizadas, estas son las que vamos a aplanar
for img_normalized in normalized_images:
    #Aplanamos la imagen y la vamos agregando a la lista creada para almacenar imágenes aplanadas, usamos una función .flatten de la librería
    flattened_images.append(img_normalized.flatten())

#convertimos la lista donde se guardaron las imagenes aplanadas a un tipo de array numpy
X_flattened = np.array(flattened_images)

#verificar las dimensiones del array resultante
print("Cantidad de imagenes a la izquierda, longitug del vector unidimensional que representa cada imagen", X_flattened.shape)

"""Una vez que tenemos las imágenes ya con todo el procesamiento completo, vamos a la parte de crear el modelo.

### División en conjunto de entrenamiento y prueba

Finalmente tenemos todos los datos ya procesados (al menos lo necesario para esta sección), así que ahora necesitamos dividir los datos en conjunto de entrenamiento y prueba.

El problema es que como vimos, tenemos un severo desbalanceo en los datos para la cantidad de las clases, recordemos cuantos datos teníamos:

- Número de imágenes en 'MildDemented': 717

- Número de imágenes en 'ModerateDemented': 52

- Número de imágenes en 'NonDemented': 2000

- Número de imágenes en 'VeryMildDemented': 1792

Es por esto que vamos a realizar una división de los datos de manera estratificada, lo que nos va a grantizar que la proporción en la división en cada clase se mantenga para ambos conjuntos de train y test de manera equilibrada, porque si no hacemos esto, al realizar la división tomando 70% de train, puede ser que todas nuestras imágenes de ModerateDemented sean seleccionadas para el Train, lo que haría que no tengamos forma de probarlas, o puede pasar al revés, por esto lo haremos de manera estratificada.

Tomaremos 70% de los datos para el conjunto de train y 30% para el test, esto porque hay una característica donde sólo tenemos 52 datos, tomar esta división nos permite obetener una cantidad considerable de datos para cada uno de los conjuntos.

Así como hicimos en el proyecto 2, lo que haremos es tomar 3 semillas: 170119, 2024 y 123456789, para observar si la división de los datos afecta la precisión del modelo, calcularemos el "mejor" modelo en base a la métrica de accuracy, usando un SVM con kernel lineal, kernel polinomial y Gaussiana, es decir, tendremos 9 combinaciones al final del cual sólo quedará una.
"""

#vamos a iterar sobre las clases que tenemos y cargar las imágenes que ya hemos preprocesado anteriormente

from sklearn.model_selection import train_test_split

#La lista de equitetas que tenemos para cada clase, las cuales ya hemos mencionado varias veces
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

#Otras dos listas más, en la lista X se guardan las imágenes procesadas, mientras que en la y se encuentran las etiquetas de cada una de las imágenes
X = []
y = []

#Otra iteración, en este caso vamos a iterar por cada etiqueta en el conjunto de etiquetas
for label in labels:
    #buscamos el directorio donde se encuentran las imágenes y obtenemos el nombre de su etiqueta
    folder_path = os.path.join(cropped_dir, label)
    #y ya que lo tenemos, iteramos sober todas las imágenes de la categoría actual
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        #cargamos ahora la imagen que ya fue normalizada
        img_normalized = io.imread(image_path)
        #Añadimos la imagen a la lista X
        X.append(img_normalized)
        #y su etiqueta a la lista y
        y.append(label)

#Convertimos nuestras listas a un array de numpy para poder realizar bien las divisiones y los demás procedimientos
X = np.array(X)
y = np.array(y)

#hacemos una división estratificada para el conjunto de train y test, en este caso usamos la semilla 170119 para observar si la división de los datos afecta
#la efectividad del modelo
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.3, random_state=170119, stratify=y)

#vamos a ver cuantos datos de cada catacterística se quedaron en cada uno de los conjuntos de los datos
#creamos una función que nos permitirá CONTAR el número de datos que hay en cada una de las etiquetas

print("\tCantidad de datos por conjuntos para la primer semilla: 170119")
print("\n")
def contar(y):
    clases, recuentos = np.unique(y, return_counts=True)
    for clase, recuento in zip(clases, recuentos):
        print(f"Cantidad de Datos en '{clase}': {recuento}")

#Calculamos el conjunto de datos de cada característica para el conjunto de train
print("Conjunto de entrenamiento:")
contar(y_train)

#y ahora lo mismo, pero para test
print("\nConjunto de prueba:")
contar(y_test)

#vamos a iterar sobre las clases que tenemos y cargar las imágenes que ya hemos preprocesado anteriormente

from sklearn.model_selection import train_test_split

#La lista de equitetas que tenemos para cada clase, las cuales ya hemos mencionado varias veces
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

#Otras dos listas más, en la lista X se guardan las imágenes procesadas, mientras que en la y se encuentran las etiquetas de cada una de las imágenes
X = []
y = []

#Otra iteración, en este caso vamos a iterar por cada etiqueta en el conjunto de etiquetas
for label in labels:
    #buscamos el directorio donde se encuentran las imágenes y obtenemos el nombre de su etiqueta
    folder_path = os.path.join(cropped_dir, label)
    #y ya que lo tenemos, iteramos sober todas las imágenes de la categoría actual
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        #cargamos ahora la imagen que ya fue normalizada
        img_normalized = io.imread(image_path)
        #Añadimos la imagen a la lista X
        X.append(img_normalized)
        #y su etiqueta a la lista y
        y.append(label)

#Convertimos nuestras listas a un array de numpy para poder realizar bien las divisiones y los demás procedimientos
X = np.array(X)
y = np.array(y)

#hacemos una división estratificada para el conjunto de train y test, en este caso usamos la semilla 170119 para observar si la división de los datos afecta
#la efectividad del modelo
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.3, random_state=2024, stratify=y)

#vamos a ver cuantos datos de cada catacterística se quedaron en cada uno de los conjuntos de los datos
#creamos una función que nos permitirá CONTAR el número de datos que hay en cada una de las etiquetas

print("\tCantidad de datos por conjuntos para la primer semilla: 2024")
print("\n")
def contar(y):
    clases, recuentos = np.unique(y, return_counts=True)
    for clase, recuento in zip(clases, recuentos):
        print(f"Cantidad de Datos en '{clase}': {recuento}")

#Calculamos el conjunto de datos de cada característica para el conjunto de train
print("Conjunto de entrenamiento:")
contar(y_train)

#y ahora lo mismo, pero para test
print("\nConjunto de prueba:")
contar(y_test)

#vamos a iterar sobre las clases que tenemos y cargar las imágenes que ya hemos preprocesado anteriormente

from sklearn.model_selection import train_test_split

#La lista de equitetas que tenemos para cada clase, las cuales ya hemos mencionado varias veces
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

#Otras dos listas más, en la lista X se guardan las imágenes procesadas, mientras que en la y se encuentran las etiquetas de cada una de las imágenes
X = []
y = []

#Otra iteración, en este caso vamos a iterar por cada etiqueta en el conjunto de etiquetas
for label in labels:
    #buscamos el directorio donde se encuentran las imágenes y obtenemos el nombre de su etiqueta
    folder_path = os.path.join(cropped_dir, label)
    #y ya que lo tenemos, iteramos sober todas las imágenes de la categoría actual
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        #cargamos ahora la imagen que ya fue normalizada
        img_normalized = io.imread(image_path)
        #Añadimos la imagen a la lista X
        X.append(img_normalized)
        #y su etiqueta a la lista y
        y.append(label)

#Convertimos nuestras listas a un array de numpy para poder realizar bien las divisiones y los demás procedimientos
X = np.array(X)
y = np.array(y)

#hacemos una división estratificada para el conjunto de train y test, en este caso usamos la semilla 170119 para observar si la división de los datos afecta
#la efectividad del modelo
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.3, random_state=123456789, stratify=y)

#vamos a ver cuantos datos de cada catacterística se quedaron en cada uno de los conjuntos de los datos
#creamos una función que nos permitirá CONTAR el número de datos que hay en cada una de las etiquetas

print("\tCantidad de datos por conjuntos para la primer semilla: 123456789")
print("\n")
def contar(y):
    clases, recuentos = np.unique(y, return_counts=True)
    for clase, recuento in zip(clases, recuentos):
        print(f"Cantidad de Datos en '{clase}': {recuento}")

#Calculamos el conjunto de datos de cada característica para el conjunto de train
print("Conjunto de entrenamiento:")
contar(y_train)

#y ahora lo mismo, pero para test
print("\nConjunto de prueba:")
contar(y_test)

