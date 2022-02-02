# Generar los Clusters
import fasttext
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from timeit import default_timer as timer

# cargar el modelo MX.bin de FastText
ft_mx = fasttext.load_model('../ft_data/MX.bin')

words = ft_mx.get_words() # todas las palabras del mx ~500,000

# words = random.sample(words, 1000) # sample
# cuando se usa una muestra pequenia de palabras,
# el resultado de los predicciones es una similitud
# entre la palabras clara y puramente lexica.
# Pero cuando se usa una muestra suficientemente grande o completa,
# las similitudes son ampliamente semanticas

# un vector de dim 300 por cada palabra
vectors = [ft_mx.get_word_vector(w) for w in words]

def minmax_normalize(vec):
    # normaliza todas las escalas de los vectores a valores entre 0 y 1
    return (vec - vec.min()) / ( vec.max() - vec.min())

def normalize(vector):
    # normaliza pero No en escala de 0 a 1
    # no lo usee
    return vector/np.linalg.norm(vector)

# normalizar los vectores a escala de 0 a 1
vectors_normalized = [minmax_normalize(v) for v in vectors]

# Clustering con k-Means. MiniBatch lo hace mas rapido
# 10,000 clusters/centroides
kmeans_mini = MiniBatchKMeans(n_clusters = 10000, init = 'k-means++', random_state = 42,  max_iter=100, verbose=1)

# El fit recibe un arreglo de Vectores; es decir, genera los clusters de vectores de 300 valores
# y no de puntos de solo 2 coordenadas x, y

start = timer()
# entrenar k-means con los vectores normalizados
kmeans_mini.fit(vectors_normalized)

print('> ', timer() - start)

if False:
    # guardar el modelo entrenado
    save_model(kmeans_mini, 'kmeans_mini_10000.obj')

# Ejemplo de Prediccion de una palabra
# Como los Clusters son de Vectores, tambien hay que predecir con vectores
# y el resultado es el cluster a donde pertenece la palabra
# Obtengo el vector de la palabra a predicir del modelo del MX.bin
# y normalizo el vector para estar en la misma escala que los clusters
vect_to_p =[minmax_normalize(ft_mx.get_word_vector('maricon'))]
# prediccion
print(kmeans_mini.predict(vect_to_p))

#kmeans_mini.cluster_centers_
#kmeans_mini.labels_
#np.where(kmeans_mini.labels_ == 10)[0]
# print(np.argmax(np.bincount(kmeans_mini.labels_))) # item most frequent
# np.bincount(kmeans_mini.labels_).argmin()
# np.sort(kmeans_mini.labels_)
