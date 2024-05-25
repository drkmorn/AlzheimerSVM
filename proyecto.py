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

base_dir = "/content/ia-2024/imgs"

#En este directorio donde se guardarán las imágenes recortadas, que se llama como el de imgs, pero cropped que significa recortado en inglés
cropped_dir = "/content/ia-2024/imgs_cropped"
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
cropped_dir = "/content/ia-2024/imgs_cropped"

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