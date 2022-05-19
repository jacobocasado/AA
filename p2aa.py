# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante:  Jacobo Casado de Gracia.
Correo electrónico en caso de dudas, consultas o posibles fallos en la ejecución.: casadojacobo@gmail.com / jacobocasado@correo.ugr.es
Código funcional creado en Python 3.9.12 y Spyder 5.1.5
"""

# Librerías necesarias para el desarrollo de la práctica.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Fijamos la semilla para los procesos pseudoaleatorios.
seed = 1
np.random.seed(seed)

# Método para pausar la ejecución en cada apartado.
# Si Spyder da un error, seguramente sea por el comando input.
# Descomentar en ese caso.
def wait():
     input("(Pulsa [Enter] para pasar al siguiente apartado...)\n")
     plt.close()

# Función que devuelve 1 si el elemento es positivo y -1 si es negativo
def sign(x):
	if x >= 0:
		return 1
	return -1

# Funciones necesarias para el ejercicio 1, proporcionadas por el profesorado.
# Sirven para generar puntos en una distribución gaussiana o uniforme.
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gauss(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

# Función que, dado un vector de pesos w y un punto x, devuelve
# el hiperplano que pasa por x.
def hiperplano(w, x):
    # A es w2, B es w1, C es w0
    A = w[2]
    B = w[1]
    C = w[0]
    
    return ((-1*B*x)/A - C/A)


#%%

# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente.
# Para ello usamos las dos funciones proporcionadas por el profesorado.
# Primero, usamos la distribución uniforme.

# Definimos los puntos a plotear.
puntos  = 50

x = simula_unif(puntos, 2, [-50,50])
x1 = x[:,0]
x2 = x[:,1]

# Ploteamos los datos en el cuadrado
plt.scatter(x1, x2, c = 'k', s=4)
plt.title(str(puntos) + ' puntos generados aleatoriamente entre los ejes del cuadrado [-50, 50] \n usando la función simula_unif')

# Damos título a los ejes del gráfico y mostramos la leyenda
plt.xlabel("Valor en x de los puntos")
plt.ylabel("Valor en y de los puntos")
plt.show()

# Hacemos lo mismo pero con la distribución gaussiana.
x = simula_gauss(puntos, 2, [5,7])
x1 = x[:,0]
x2 = x[:,1]

# Ploteamos los datos en el cuadrado
plt.scatter(x1, x2, c = 'k', s=4)
plt.title(str(puntos) + ' puntos generados aleatoriamente entre los ejes del cuadrado [5, 7] \n usando la función simula_gauss')

# Damos título a los ejes del gráfico y mostramos la leyenda
plt.xlabel("Valor en x de los puntos")
plt.ylabel("Valor en y de los puntos")
plt.show()

wait()


#%%

# EJERCICIO 1.2: Valorar la influencia del ruido.
# MÉTODOS NECESARIOS para la ejecución de este ejercicio.

# Función que devuelve los coeficientes (a,b) de una recta 2D que pasa por los extremos
# de un intervalo.
def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

# La funcion np.sign(0) da 0, lo que nos puede dar problemas,
def signo(x):
	if x >= 0:
		return 1
	return -1

# Función que etiqueta los puntos X dada su posición respecto de una recta
# con los coeficientes (a,b). Etiqueta como +1, o -1, al usar el signo.
def f(x, y, a, b):
	return signo(y - a*x - b)

# Función que aplica ruido a un percentage de puntos de y.
# Aplica directamente el ruido. No devuelve nada.
def apply_noise(y, percentage):
    
    # Seleccionamos los índices de las clases positivas (resp. negativas)
    idxs_negative = np.where(y == -1)[0]
    idxs_positive = np.where(y == 1)[0]

    # Elegimos aleatoriamente una fracción 'p' de ellos
    random_idxs_negative = np.random.choice(idxs_negative, round(percentage * len(idxs_negative)), replace = False)
    random_idxs_positive = np.random.choice(idxs_positive, round(percentage * len(idxs_positive)), replace = False)
    # Cambiamos el signo de los elegidos
    y[random_idxs_negative] = -y[random_idxs_negative]
    y[random_idxs_positive] = -y[random_idxs_positive]
    
 
# Función dada por el profesorado que imprime los gráficos para
# el apartado 2.c)       
def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('Valor de las etiquetas')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=1, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
# Función que,dado un conjunto de puntos y una recta (a,b), 
# cuenta el tanto por uno de aciertos.
def get_accuracy_recta(x, a,b, y):
    x1 = x[:,0]
    x2 = x[:,1]
    y_pred = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])
    
    ein = np.count_nonzero(y_pred-y)/np.shape(y)[0]
    print("Error de clasificación binario (tanto por uno de fallos): ", ein)
    

#%%

# EJERCICIO 1.2: Valorar la influencia del ruido.

# Para el apartado a), se comienzan generando 100 puntos y etiquetándolos
# con  una recta que pasa por el intervalo definido por los puntos.

# definimos el intervalo
intervalo = [-50,50]
# los puntos sin etiquetar distribuidos de manera uniforme
x = simula_unif(100, 2, intervalo)
x1 = x[:,0]
x2 = x[:,1]
# creamos la recta que pasa por el intervalo
a,b = simula_recta(intervalo)
# etiquetamos los puntos con esta recta
y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])
# comprobamos la precisión de clasificado con esta misma recta
print("Apartado 1.2.a) Clasificación sin ruido.")
get_accuracy_recta(x, a,b,y)

# Graficamos los datos con sus etiquetas, incluyendo en la leyenda
# de qué clase son.
data_negative = np.array([x_i for x_i, y_i in zip(x,y) if y_i == -1])
data_positive = np.array([x_i for x_i, y_i in zip(x,y) if y_i == 1])
plt.scatter(data_negative[:,0], data_negative[:,1], c='red', label = 'Clase -1', s = 5)
plt.scatter(data_positive[:,0], data_positive[:,1], c='blue', label = 'Clase 1',  s = 5)

# Graficamos la recta además de la nube de puntos
recta_x = np.linspace(-50,50,100)
recta_y = a*recta_x+b
formatted_a = '{0:.2f}'.format(a)
formatted_b = '{0:.2f}'.format(b)
plt.plot(recta_x, recta_y, '-k', label='f(x,y) = y - ax - b')

# Configuración adicional del plot.
plt.xlim(intervalo)
plt.ylim(intervalo)
plt.title('100 puntos etiquetados por la recta f(x,y) = y - ax - b \n a = ' + str(formatted_a) + '; b = ' + str(formatted_b))
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend(loc = 1)
plt.show()

wait()

#%%

# Para el apartado b), hacemos EXACTAMENTE LO MISMO
# pero llamamos a esta función para crear un 10% de ruido en la distribución 
# de puntos.
apply_noise(y, 0.1)
# Comprobamos el tanto por uno de la clasificación con ruido.
print("Apartado 1.2.b) Clasificación CON ruido.")
get_accuracy_recta(x, a,b,y)

# Graficamos los datos con sus etiquetas, incluyendo en la leyenda
# de qué clase son.
data_negative = np.array([x_i for x_i, y_i in zip(x,y) if y_i == -1])
data_positive = np.array([x_i for x_i, y_i in zip(x,y) if y_i == 1])
plt.scatter(data_negative[:,0], data_negative[:,1], c='red', label = 'Clase -1', s = 5)
plt.scatter(data_positive[:,0], data_positive[:,1], c='blue', label = 'Clase 1',  s = 5)

# Graficamos la recta además de la nube de puntos
recta_x = np.linspace(-50,50,100)
recta_y = a*recta_x+b
formatted_a = '{0:.2f}'.format(a)
formatted_b = '{0:.2f}'.format(b)
plt.plot(recta_x, recta_y, '-k', label='f(x,y) = y - ax - b')

# Configuración adicional del plot.
plt.xlim(intervalo)
plt.ylim(intervalo)
plt.title('100 puntos CON RUIDO etiquetados por la recta f(x,y) = y - ax - b \n a = ' + str(formatted_a) + '; b = ' + str(formatted_b))
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend(loc = 1)
plt.show()

wait()
#%%

# Para el apartado c) de este ejercicio 1.2, hace falta 
# diseñar 4 funciones auxiliares:
    
def f1(X):
    return (X[:,0]-10)**2 + (X[:,1]-20)**2 - 400

def f2(X):
    return 0.5 * (X[:,0]+10)**2 + (X[:,1]-20)**2 - 400

def f3(X):
    return 0.5 * (X[:,0]-10)**2 - (X[:,1]+20)**2 - 400

def f4(X):
    return X[:,1]-20*X[:,0]**2-5*X[:,0]+3

# Con esta función, que recibe otra función f y un set de datos X, 
# se devuelve un vector y correspondiente de etiquetar X con f.
def etiquetar_puntos_f(f, x):
    y = np.sign(f(x))
    return y

# Con esta función se devuelve el tanto por uno de error de clasificación obtenido
# usando f para etiquetar los puntos en comparación con su valor real y.
def evaluar_f(f, x,y):

    y_hat = np.sign(f(x))

    errors = (y_hat - y) != 0

    Ein = np.mean(errors)
    print("Error de clasificación obtenido: ", Ein)
    return (Ein)
    
#%%
# Evaluación con los puntos etiquetados originalmente con la recta del ejercicio 1.

print("Apartado 2.c.1 - Evaluación con las nuevas fronteras a los puntos del ejercicio 2.b)")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)
print("--------------")

plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')

wait()

# Usamos f1 para etiquetar los puntos ahora, y comprobamos 
# el error de clasificación para las otras rectas.

y = etiquetar_puntos_f(f1, x)
apply_noise(y, 0.1)
print("Apartado 2.c.2 - Etiquetando con f1")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)
print("--------------")

# SI SE DESEA MOSTRAR EL AJUSTE, HE COMENTADO LA FUNCIÓN PORQUE SON MUCHOS
# GRÁFICOS A MOSTRAR (REALMENTE SON LAS MISMAS FUNCIONES, CAMBIAN LOS PUNTOS)
# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')


# Usamos f2 para etiquetar los puntos ahora, y comprobamos 
# el error de clasificación para las otras rectas.
y = etiquetar_puntos_f(f2, x)
apply_noise(y, 0.1)
print("Apartado 2.c.2 - Etiquetando con f2")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)
print("--------------")


# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')

# Usamos f3 para etiquetar los puntos ahora, y comprobamos 
# el error de clasificación para las otras rectas.
y = etiquetar_puntos_f(f3, x)
apply_noise(y, 0.1)
print("Apartado 2.c.2 - Etiquetando con f3")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)
print("--------------")

# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')


# Usamos f3 para etiquetar los puntos ahora, y comprobamos 
# el error de clasificación para las otras rectas.
y = etiquetar_puntos_f(f4, x)
apply_noise(y, 0.1)
print("Apartado 2.c.2 - Etiquetando con f4")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)
print("--------------")

# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')

wait()

#%%
# EJERCICIO 2. MODELOS LINEALES.

# Funciones generales para el apartado del PLA y de RL.

# Función que añade un bias de 1 al array X de datos.
# Necesario para los métodos de PLA y RL, que necesitan un bias para su funcionamiento.
def add_bias(x):
    x1 = x[:,0]
    x2 = x[:,1] 
    x_bias = np.array([[1,x1,x2] for x1, x2 in zip(x1,x2)])
    return (x_bias)

# Función que, dado un vector de puntos X con bias, un vector de pesos
# solución w, y un vector de etiquetas Y, comprueba el tanto por uno de aciertos
# que ha realizado el hiperplano formado por w sobre los datos X en comparación
# a sus etiquetas reales Y.
def get_accuracy(w, x_bias, y):
    
    errors = 0
    hits = 0

    # Recorremos todos los ejemplos
    for x, l in zip(x_bias, y):
        # Si está mal clasificado, actualizamos los pesos
        if sign(x.dot(w)) != l:
            errors = errors + 1
        else:
            hits = hits + 1
            
    #print ("Errores: ", errors, "\n Aciertos : ", hits)
    #print ("Ein: ", (errors / (errors + hits)))
    #print ("Accuracy: ", (hits / (errors + hits)))
    
    # Tanto por uno de HITS.
    accuracy = (hits / (errors + hits))
    #print(accuracy)
    
    return accuracy


# Función que, dado un vector de puntos X con bias, un vector de pesos
# solución w, y un vector de etiquetas Y, comprueba el tanto por uno de FALLOS
# que ha realizado el hiperplano formado por w sobre los datos X en comparación
# a sus etiquetas reales Y.
# Realmente es la analogía a la función de arriba pero con fallos.
def get_classification_error(w, x_bias, y):
    
    errors = 0
    hits = 0

    # Recorremos todos los ejemplos
    for x, l in zip(x_bias, y):
        # Si está mal clasificado, actualizamos los pesos
        if sign(x.dot(w)) != l:
            errors = errors + 1
        else:
            hits = hits + 1
            
    #print ("Errores: ", errors, "\n Aciertos : ", hits)
    #print ("Ein: ", (errors / (errors + hits)))
    #print ("Accuracy: ", (hits / (errors + hits)))
    
    ein = (errors / (errors + hits))
    #print(accuracy)
    
    return ein

#%%

# EJERCICIO 2.1: ALGORITMO PERCEPTRON
# Funciones necesarias para el desarrollo de este apartado.

""" Ejercicio del algoritmo PLA (perceptrón).
Recibe:
    Conjunto de datos X con bias incluido.
    Vector de etiquetas Y (label)
    Criterio alternativo de parada: máximo de iteraciones (max_iter).
    Vini: vector de pesos inicial.
El algoritmo calcula el vector de pesos w final y devuelve:
    El vector de pesos w final
    La iteración en la que ha convergido.
    El porcentaje de aciertos y de fallos con este vector de pesos solución.
La explicación teórica se encuentra en la memoria.
"""
def ajusta_PLA(datos_bias, label, max_iter, vini):
        
    w = vini.copy()  # No modificamos el parámetro w_ini
    evol = [vini] # Guardaremos la evolución de los w, para calcular la accuracy.

    # Repetimos el algoritmo el número de iteraciones deseado
    for it in range(max_iter + 1):
        change = False

        # Recorremos todos los ejemplos
        for x, l in zip(datos_bias, label):
            # Si está mal clasificado, actualizamos los pesos
            if sign(x.dot(w)) != l:
                w += l * x
                change = True

        # Si no ha habido cambios, terminamos.
        if not change:
            break
        
        # Guardamos la evolución de w
        evol.append(w.copy())
        
    # Imprimimos los resultados antes de devolverlos.
    print("-w final: ", w, "\n", "-Iteración final: ", it, "\n", "-Error de clasificación: ", get_classification_error(w, datos_bias, label))

    return w, it, evol, get_accuracy(w, datos_bias, y), get_classification_error(w, datos_bias, label)

""" Variante del algoritmo PLA (perceptrón) llamada PLA-POCKET.
    Su comportamiento es igual al perceptrón, pero se almacena el mejor
    vector de pesos w (w_best) de manera que se guarda la mejor solución.
Recibe:
    Conjunto de datos X con bias incluido.
    Vector de etiquetas Y (label)
    Criterio alternativo de parada: máximo de iteraciones (max_iter).
    Vini: vector de pesos inicial.
El algoritmo calcula el vector de pesos w final y devuelve:
    El vector de pesos w final CON MENOR ERROR DE CLASIFICACIÓN (w_best)
    Una traza de la evolución de w_best.
    El porcentaje de aciertos y de fallos con este vector de pesos solución best.
La explicación teórica se encuentra en la memoria.
"""
def pla_pocket(datos_bias, label, max_iter, vini):
    
    w = vini.copy() # No modificamos el original
    w_best = w.copy() # Almacenamos EL MEJOR.
    # Necesitamos el error de clasificación del MEJOR.
    best_err = get_classification_error(w_best, datos_bias, label) 
    # Guardamos la traza de w_best.
    evol = [vini]

    for _ in range(max_iter):
        
        for x, l in zip(datos_bias, label):
            if sign(x.dot(w)) != l:
                w += l * x
            # Criterio de sustitución de w_best:
            # Encontrar un vector de pesos con menor error que el mejor.
            curr_err = get_classification_error(w, datos_bias, label)
            # Si el actual es mejor que w_best en términos de clasificación,
            # Lo reemplaza.
            if curr_err < best_err:
                best_err = curr_err
                w_best = w.copy()
        # Almacenamos w_best en la traza cada iteración.
        evol.append(w_best.copy())
    
    # Imprimimos antes de devolver.
    print("-w final: ", w_best, "\n", "-Iteración final: ", max_iter, "\n -Error de clasificación: ", get_classification_error(w_best, datos_bias, label))

    return w_best, evol, get_classification_error(w_best, datos_bias, label)
        
# Método que, dada una traza de w, grafica la evolución de la accuracy de esa traza
# sobre un conjunto de datos X.
def plot_accuracy(evol, x, y):
    # Mostramos una gráfica con la evolución del accuracy
    acc_evol = []
    for w_ in evol:
        acc_evol.append(get_accuracy(w_, add_bias(x), y))
        
    plt.figure(figsize = (8, 6))
    plt.xlabel("Iteraciones")
    plt.ylabel("Accuracy")
    plt.title("Evolución del accuracy en la clasificación durante el algoritmo.")
    plt.plot(range(len(evol)), acc_evol)
    plt.show()
    
# Método que, dada una traza de w, grafica la evolución del error de clasificación de esa traza
# (tanto por uno de fallos, al ser perceptrón) sobre un conjunto de datos X.
def plot_ein(evol, x, y):
    # Mostramos una gráfica con la evolución del accuracy
    acc_evol = []
    for w_ in evol:
        acc_evol.append(get_classification_error(w_, add_bias(x), y))
        
    plt.figure(figsize = (8, 6))
    plt.xlabel("Iteraciones")
    plt.ylabel("Ein")
    plt.title("Evolución del Ein (error de clasificación, tanto por uno) \n en la clasificación durante el algoritmo.")
    plt.plot(range(len(evol)), acc_evol)
    plt.show()

# Método que, dado un vector de pesos w, grafica LA FRONTERA DE DECISIÓN GENERADA
# por ese vector de pesos y la nube de puntos X con su etiqueta Y.
# Sirve para ver el ajuste de un set de hipótesis de manera visual.
def plot_recta(x,y,intervalo,w):
    
    # Generamos la recta y la ploteamos usando w.
    a,b = simula_recta(intervalo)
    x1_recta = hiperplano(w, intervalo[0])
    x2_recta = hiperplano(w, intervalo[1])
    recta = ([x1_recta, x2_recta])
    plt.plot(intervalo, recta, '-k', label='f(x,y) = y - ax - b')
    
    # Graficamos los datos con sus etiquetas, incluyendo en la leyenda
    # de qué clase son.
    data_negative = np.array([x_i for x_i, y_i in zip(x,y) if y_i == -1])
    data_positive = np.array([x_i for x_i, y_i in zip(x,y) if y_i == 1])
    plt.scatter(data_negative[:,0], data_negative[:,1], c='red', label = 'Clase -1', s = 5)
    plt.scatter(data_positive[:,0], data_positive[:,1], c='blue', label = 'Clase 1',  s = 5)
    
    # Configuración adicional del ploteo.
    plt.xlim(intervalo)
    plt.ylim(intervalo)
    plt.title('Ajuste de la recta f(x,y) = y - ax - b sobre 100 puntos. \n a = ' + str(formatted_a) + '; b = ' + str(formatted_b))
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.legend(loc = 1)
    plt.show()
    

#%%
# Ejecutamos el algoritmo PLA con los datos del ejercicio 2.1.a
# ES DECIR, SIN RUIDO.
print("Ejercicio 2.1.a): Perceptrón SIN RUIDO")

# Definimos el intervalo
intervalo = [-50,50]
x = simula_unif(100, 2, intervalo)
# Generamos los puntos en el intervalo dado.
x1 = x[:,0]
x2 = x[:,1]
# Etiquetamos los puntos con una recta arbitraria
y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])

# Primero, lo ejecutamos con el [0,0,0] como pide el apartado.
print (" \n Ejecutando con w_ini = [0,0,0]. Resultados:")
w, it, evol, acc, ein = ajusta_PLA(add_bias(x), y, 1000, [0.,0.,0.])

# Dibujamos la recta original, la que etiqueta esos puntos.
recta_x = np.linspace(intervalo[0], intervalo[1],100)
recta_y = a*recta_x+b
plt.plot(recta_x, recta_y, '--b', label='Recta etiquetado')
plt.legend(loc='upper right')

# Dibujamos el ajuste de la recta VS el original, y el avance de la accuracy y el error.
plot_recta(x,y,intervalo,w)
plot_accuracy(evol, x, y)
plot_ein(evol, x, y)

wait()

# Luego lo ejecutamos con vectores de números aleatorios en [0,1] 10 veces
# Y hacemos la media de las iteraciones.
# Imprimimos la información pedida en el enunciado.
iterations = []
error = []
for i in range(0,10):
    w_random =  np.random.rand(3)
    print("Ejecutando PLA con w inicial: ", w_random)
    w, it, evol, acc, ein  = ajusta_PLA(add_bias(x), y, 1000, w_random)
    iterations.append(it)
    error.append(ein)
    print ("---")

# Imprimimos los resultados medios después de las 10 ejecuciones.
print (" \n Fin de las 10 ejecuciones. Resultados:")
print ("Iteración media después de las 10 ejecuciones: ", np.mean(iterations))
print ("Error medio de clasificación después de las 10 ejecuciones: ", np.mean(ein))

wait()

#%%
# Ahora, en el ejercicio 2.1.b, hacemos lo mismo pero añadimos ruido.

print("Ejercicio 2.1.a): Perceptrón CON RUIDO")
y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])
# Aplicamos ruido a este vector y, un 10% de los datos.
apply_noise(y, 0.1)

# Primero, lo ejecutamos con el [0,0,0]
print (" \n Ejecutando con w_ini = [0,0,0]. Resultados:")
w, it, evol, acc, ein = ajusta_PLA(add_bias(x), y, 1000, [0.,0.,0.])

# Dibujamos la recta original, la que etiqueta esos puntos.
recta_x = np.linspace(intervalo[0], intervalo[1],100)
recta_y = a*recta_x+b
plt.plot(recta_x, recta_y, '--b', label='Recta etiquetado')
plt.legend(loc='upper right')

# Dibujamos el ajuste de la recta VS el original, y el avance de la accuracy y el error.
plot_recta(x,y,intervalo,w)
plot_accuracy(evol, x, y)
plot_ein(evol, x, y)

wait()

# Luego lo ejecutamos con vectores de números aleatorios en [0,1] 10 veces
# Y hacemos la media de las iteraciones.
# Imprimimos la información pedida en el enunciado.
iterations = []
error = []
for i in range(0,10):
    w_random =  np.random.rand(3)
    print("Ejecutando PLA con w inicial: ", w_random)
    w, it, evol, acc, ein  = ajusta_PLA(add_bias(x), y, 1000, w_random)
    iterations.append(it)
    error.append(ein)
    print ("---")

# Imprimimos los resultados medios después de las 10 ejecuciones.
print (" \n Fin de las 10 ejecuciones. Resultados:")
print ("Iteración media después de las 10 ejecuciones: ", np.mean(iterations))
print ("Error medio de clasificación después de las 10 ejecuciones: ", np.mean(ein))

wait()

#%%

print("Ejercicio 2.1.a): PLA POCKET CON RUIDO (OPCIONAL)")

# Probamos a ajustar los datos con ruido con el PLA Pocket.

# Probamos con el [0,0,0] para hacer las gráficas.
w, evol, ein = pla_pocket(add_bias(x), y, 100, [0.,0.,0.])

# Dibujamos la recta original, la que etiqueta esos puntos.
recta_x = np.linspace(intervalo[0], intervalo[1],100)
recta_y = a*recta_x+b
plt.plot(recta_x, recta_y, '--b', label='Recta etiquetado')
plt.legend(loc='upper right')

# Dibujamos el ajuste de la recta VS el original, y el avance de la accuracy y el error.
plot_recta(x,y,intervalo,w)
plot_accuracy(evol, x, y)
plot_ein(evol, x, y)

# Luego probamos con 10 inicializaciones en aleatorio.
iterations = []
error = []
for i in range(0,10):
    w_random =  np.random.rand(3)
    print("Ejecutando PLA POCKET con w inicial: ", w_random)
    w, evol, ein = pla_pocket(add_bias(x), y, 100, [0.,0.,0.])
    error.append(ein)
    print ("---")
    
print ("Error medio de clasificación después de las 10 ejecuciones: ", np.mean(ein))

wait()

#%%
# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
# Para este apartado, necesitamos definir la función de error 
# de entropía cruzada y su gradiente.
def err(x, y, w):
    """Expresión del error cometido por un modelo de regresión logística
        (entropía cruzada)
         - X: matriz de datos con bias.
         - y: vector de etiquetas.
         - w: vector de pesos."""
         
    # Trasponemos w.
    w_t = np.transpose(w)
    # Realizamos la fórmula.
    return np.mean(np.log(1 + np.exp(-y * x.dot(w_t))))

def derr(x, y, w):
    """Expresión puntual del gradiente del error cometido por un modelo
       de regresión logística (entropía cruzada.)
         - x: vector de características con primera componente 1.
         - y: vector de etiquetas.
         - w: vector de pesos."""
    # Trasponemos w.     
    w_t = np.transpose(w)
    # Calculamos el gradiente de todos los puntos del batch, que se han pasado
    # como parámetro
    grad = np.array([ (y[i] * x[i]) / (1 + np.exp(y[i] * np.dot(w_t, x[i]))) for i in range (len(y))])
    # Hacemos la media de este gradiente.
    return -np.mean(grad, axis = (0))

# Función que obtiene el batch i de un conjunto de datos X con sus etiquetas Y.
# Sirve para obtener batches en el algoritmo de SGD.
def get_batches(x, y, i, batch_size):
    
    # Obtenemos el batch i de datos.
    x_batch = x[i:i+batch_size]
    # Así como de sus etiquetas.
    y_batch = y[i:i+batch_size]
    
    # Devolvemos ambos batches.
    return x_batch, y_batch


# Función ya definida anteriormente pero se redefine aquí para mejor claridad.
# Etiqueta los puntos X dada la recta (a,b) (devuelve el signo)
def f(x, y, a, b):
	return signo(y - a*x - b)

def logistic_sgd(x, y, lr, eps, batch_size, verbose = 0):
    """Función que implementa el algoritmo de regresión logística con SGD. 
    Recibe:
        - X: matriz de datos, cada uno primera componente 1.
        - y: vector de etiquetas.
        - lr: valor del learning rate.
        - eps: tolerancia para el criterio de parada.
        - batch size: tamaño de batch para actualizar w en cada paso.
        - opción para que imprima o no los resultados. Por defecto, no los imprime.
    Devuelve:
         - vector de pesos encontrado y las iteraciones realizadas, así como
             el error de entropía cruzada (Ein)y el error de clasificación
             en tanto por uno.
     """
    
    # Creamo el vector w que tendrá 0 en todas sus posiciones, del tamaño
    # de los datos (con el bias)
    d = len(x[0])
    w = np.zeros(d)  # Establecemos el punto inicial.
    
    # Inicializamos las iteraciones a 0, y el criterio de parada a falso.
    it = 0
    converged = False
    
    # Mientras que no se cumpla la condición de parada:
    while not converged:
        # No modificamos w.
        w_old = w.copy()

        # Barajamos los índices y actualizamos los pesos por cada iteración.
        x, y = shuffle(x, y, random_state=seed)
        
        # Recorremos los batches (SGD)
        for i in range(0, len(x), batch_size):
            # Obtenemos el batch i de datos y etiquetas.
            x_batch_i, y_batch_i = get_batches(x,y,i, batch_size)
            # Actualizamos w con el gradiente del error para este batch.
            w -= lr * derr(x_batch_i, y_batch_i, w)

        # Comprobamos condición de parada
        converged = np.linalg.norm(w_old - w) < eps
        it += 1
    
    # Cuando finaliza, calculamos Ein y el tanto por uno de error de clasificación.
    error = err(x,y,w)
    class_error = get_classification_error(w, x, y)
    
    if verbose == 1:
        print (" -w inicial: ", w_old)
        print(" -w final: ", w, "\n", "-Iteración final: ", it, "\n -Error Ein: ", error, "\n -Error de clasificación: ", class_error)

    return w, it, error,  class_error


#%%

print("Ejercicio 2.2 - Regresión Logística")

# Definimos un intervalo
intervalo = [0,2]
# Creamos una recta arbitraria que pasa por el intervalo.
a,b = simula_recta(intervalo)
# Seleccionamos 100 puntos aleatorios Xn de X.
x = simula_unif(100, 2, intervalo)
x1 = x[:,0]
x2 = x[:,1]
# Evaluamos las respuestas yn de todos ellos respecto a la frontera elegida.
y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])
# Ejecutamos RL para encontrar g
w, it, Ein, class_error = logistic_sgd(add_bias(x), y, 1, 0.01, 32, verbose = 1)
# Dibujamos la recta original, la que etiqueta esos puntos.
recta_x = np.linspace(-50,50,100)
recta_y = a*recta_x+b
plt.plot(recta_x, recta_y, '--b', label='Recta etiquetado')
plt.legend(loc='upper right')
plot_recta(x,y,intervalo,w)

wait()

#%%

# Probamos a graficar el ajuste ejecutando RL una sola vez con datos de test.
# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).
print("Ejercicio 2.2 - Probando con un conjunto de test de 1000 nuevos datos.")

# Seleccionamos 1000 puntos aleatorios Xn de X, como conjunto de test.
x_test = simula_unif(1000, 2, intervalo)
x1 = x_test[:,0]
x2 = x_test[:,1]
# Evaluamos las respuestas yn de todos ellos respecto a la frontera elegida
# para obtener sus etiquetas reales.
y_test = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])

# Calculamos el error en test de nuestro algoritmo RL.
Eout = err(add_bias(x_test), y_test, w)
print (" -Eout: ", Eout)
print (" -Error de clasificación con el conjunto test: ", get_classification_error(w, add_bias(x_test), y_test))

# Graficamos la recta original
recta_x = np.linspace(-50,50,100)
recta_y = a*recta_x+b
plt.plot(recta_x, recta_y, '--b', label='Recta etiquetado')
plt.legend(loc='upper right')

# Graficamos los 1000 puntos y la recta que ha generado nuestro algoritmo.
plot_recta(x_test,y_test,intervalo,w)

wait()

#%%

print("Ejercicio 2.2 - Ejecutando RL 100 veces:")


# Almacenamos las métricas para posteriormente hacer su promedio.
Eout_promedio = []
Ein_promedio = []
iteraciones_promedio = []
class_error_train_promedio = []
class_error_test_promedio = []

for i in range (0, 100):
    # Seleccionamos 100 puntos aleatorios Xn de X.
    x = simula_unif(100, 2, intervalo)
    x1 = x[:,0]
    x2 = x[:,1]
    # Evaluamos las respuestas yn de todos ellos respecto a la frontera elegida.
    y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])
    # Ejecutamos RL para encontrar g
    w, it, Ein, class_error = logistic_sgd(add_bias(x), y, 1, 0.01, 32)
    iteraciones_promedio.append(it)
    Ein_promedio.append(ein)
    class_error_train_promedio.append(class_error)
    # Evaluamos el error Eout usando pra ello una nueva muestra de datos (>999)
    x_test = simula_unif(1000, 2, intervalo)
    x1 = x_test[:,0]
    x2 = x_test[:,1]
    # Evaluamos las respuestas yn de todos ellos respecto a la frontera elegida.
    y_test = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])
    # Calculamos las métricas restantes usando el conjunto de test.
    Eout = err(add_bias(x_test), y_test, w)
    Eout_promedio.append(Eout)
    class_error_test_promedio.append(get_classification_error(w, add_bias(x_test), y_test))

# Calculamos el promedio de las métricas.
Eout_promedio = np.mean(Eout_promedio)
Ein_promedio = np.mean(Ein_promedio)
iteraciones_promedio = np.mean(iteraciones_promedio)
class_error_train_promedio = np.mean(class_error_train_promedio)
class_error_test_promedio = np.mean(class_error_test_promedio)

# Las imprimimos.
print("Fin del experimento con 100 ejecuciones. Métricas:")
print("Iteraciones promedio: ", iteraciones_promedio)
print("Error Ein de entrenamiento promedio: ", Ein_promedio)
print("Error de clasificación de entrenamiento promedio: ", class_error_train_promedio)
print("Error Eout de test promedio: ", Eout_promedio)
print("Error de clasificación de test promedio: ", class_error_test_promedio)
print("---")

wait()

#%%

## APARTADO DE BONUS. Clasificación de dígitos.

# Definimos el método de regresión lineal, usando la Pseudoinversa.
# No comento nada puesto que esto es materia de la práctica 1.

def pseudoinverse(X,Y):
    
    U,D,V_traspose = np.linalg.svd(X, full_matrices=False)
    D = np.diag(D)
    V = np.transpose(V_traspose)
    
    A = np.dot(V,np.linalg.inv(D))
    A = np.dot(A, np.transpose(U))
    
    w = np.dot(A, Y)
    
    return w

# Error asociado a regresión lineal (Error cuadrático medio)
def Ecm(x,y,w):
    
    w_t = np.transpose(w)
    
    predictions = np.array([np.dot(w_t, x_n) for x_n in x])
    
    error = predictions - y
    
    error_cuadratico = error * error 
    
    ecm = np.mean(error_cuadratico)
    
    return ecm


# Funcion para leer los datos, proporcionada por el profesorado.
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y


""" Función que grafica el ajuste generado por el set de hipótesis g con nuestro
    vector de pesos w, sobre los datos de la práctica.
    Únicamente recibe el vector de pesos w solución, para dibujar la frontera de
    clasificación asociada, y una etiqueta para mostrar en la leyenda, que
    dependerá del algoritmo de aprendizaje usado.
    
    La función imprime el ajuste tanto en entrenamiento como en test, en un
    solo plot en mosaico.
  """
def plot_ajuste(w, label):
    
    
    #mostramos los datos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    x1_recta = hiperplano(w, intervalo[0])
    x2_recta = hiperplano(w, intervalo[1])
    recta = ([x1_recta, x2_recta])

    for ax in fig.get_axes():
        ax.label_outer()

    ax1.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
    ax1.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
    ax1.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (ENTRENAMIENTO)')
    ax1.plot(intervalo, recta, '-k', label='Pseudoinversa')
    ax1.set_xlim((0, 0.6))
    ax1.set_ylim((-8, 0))

    ax2.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
    ax2.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
    ax2.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
    ax2.plot(intervalo, recta, '-k', label=label)

    ax2.set_xlim((0, 0.6))
    ax2.set_ylim((-8, 0))
    plt.legend()
    plt.show()
    
    wait()

    
    
#%%

print("BONUS. CLASIFICACIÓN DE DÍGITOS.")
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
print("Tamaño de train: ", np.shape(x)[0], "datos.")
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])
print("Tamaño de test: ", np.shape(x_test)[0], "datos.")

#mostramos los datos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

for ax in fig.get_axes():
    ax.label_outer()

ax1.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax1.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax1.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (ENTRENAMIENTO)')
ax1.set_xlim((0,0.6))

ax2.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax2.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax2.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax2.set_xlim((0, 0.6))
plt.legend()
plt.show()

wait()

#%%

# AJUSTE USANDO LA PSEUDOINVERSA.
print("Ajuste usando la PSEUDOINVERSA")

w = pseudoinverse(x,y)

# Calculamos los errores en entrenamiento y test (usando ECM, asociado a regresión lineal.)
Ein_pseudo = Ecm(x,y,w)
Etest_pseudo = Ecm(x_test,y_test,w)

print("Ein: (error cuadrático medio): ", Ein_pseudo)
print("Error de clasificación (tanto por uno) en TRAIN: ", get_classification_error(w,x,y))
print("Etest: (error cuadrático medio): ", Ecm(x_test,y_test,w))
print("Error de clasificación (tanto por uno) en TEST: ", get_classification_error(w,x_test,y_test))

plot_ajuste(w, 'Pseudoinversa')

#%%

# AJUSTE USANDO PLA NORMAL
print("-----------")
print("Ajuste usando PLA")
w, it, evol, acc, ein = ajusta_PLA(x, y, 100, [0.,0.,0.])

Ein_pla = get_classification_error(w,x,y)
Etest_pla = get_classification_error(w,x_test,y_test)

print("Error de clasificación (tanto por uno) en TRAIN: ", Ein_pla)
print("Error de clasificación (tanto por uno) en TEST: ", Etest_pla)

plot_ajuste(w, 'PLA Algorithm')


#%%
# AJUSTE USANDO PLA POCKET
print("-----------")
print("Ajuste usando PLA POCKET")
w, evol, ein = pla_pocket(x, y, 10, [0.,0.,0.])

Ein_pocket = get_classification_error(w,x,y)
Etest_pocket = get_classification_error(w,x_test,y_test)

print("Error de clasificación (tanto por uno) en TRAIN: ", Ein_pocket)
print("Error de clasificación (tanto por uno) en TEST: ", Etest_pocket)

plot_ajuste(w, 'PLA Pocket')

#%%

# AJUSTE USANDO REGRESIÓN LOGÍSTICA
print("-----------")
print("Ajuste usando REGRESIÓN LOGÍSTICA")
w, it, Ein, class_error = logistic_sgd(x, y, 1, 0.01, 32)

Ein_RL = err(x,y,w)
Etest_RL =  err(x_test,y_test,w)

print("Ein: (error cross-entropy): ",Ein_RL)
print("Error de clasificación (tanto por uno) en TRAIN: ", get_classification_error(w,x,y))
print("Etest: (error cross-entropy): ", Etest_RL)
print("Error de clasificación (tanto por uno) en TEST: ", get_classification_error(w,x_test,y_test))

plot_ajuste(w, 'Regresión Logística')

#%%

# Bonus apartado 3.
# Probamos a generar w con pseudoinversa y se lo pasamos a PLA Pocket.

w_pseudo = pseudoinverse(x,y)

w, evol, ein = pla_pocket(x, y, 10, w_pseudo)
print("Error de clasificación (tanto por uno) en TRAIN: ", get_classification_error(w,x,y))
print("Error de clasificación (tanto por uno) en TEST: ", get_classification_error(w,x_test,y_test))

# PLA BÁSICO
w, it, evol, acc, ein = ajusta_PLA(x, y, 100, w_pseudo)
print("Error de clasificación (tanto por uno) en TRAIN: ", get_classification_error(w,x,y))
print("Error de clasificación (tanto por uno) en TEST: ", get_classification_error(w,x_test,y_test))

# Vemos que el error es el mismo en la memoria.

wait()

#%%

# Bonus apartado 4. cota sobre el error.
# Definimos ambas cotas, la cota VC y la cota de HOEFFDING,
# cuya explicación está en la memoria.

def err_bound_hoeffding(err, n, m, delta):
    """Cota de Hoeffding para el error de generalización.
         - err: error a partir del cual generalizamos.
         - n: tamaño del conjunto usado para calcular 'err'.
         - m: tamaño de la clase de hipótesis usada para calcular 'err'.
         - delta: tolerancia."""

    return err + np.sqrt((1 / (2 * n)) * np.log((2 * m) / delta))

def err_bound_vc(err, n, vc, delta):
    """Cota para el error de generalización basada en la dimensión VC.
         - err: error a partir del cual generalizamos.
         - n: tamaño del conjunto usado para calcular 'err'.
         - vc: dimensión VC del clasificador usado para calcular 'err'.
         - delta: tolerancia."""

    return err + np.sqrt((8 / n) * np.log(4 * ((2 * n) ** vc + 1) / delta))

#%%

# Fijamos el delta.
delta = 0.05

# Imprimimos las cotas de cada algoritmo usando los dos criterios.
print ("PSEUDOINVERSA: Eout como máximo vale, con una probabilidad de ", str(1 - delta), ":")
print("Cota usando Ein (dimensión VC): ", err_bound_vc(Ein_pseudo, np.shape(x)[0], 3, delta))
print("Cota usando Etest (Hoeffding): ", err_bound_hoeffding(Etest_pseudo, np.shape(x_test)[0], 1, delta))

print ("PLA: Eout como máximo vale, con una probabilidad de ", str(1 - delta), ":")
print("Cota usando Ein (dimensión VC): Eout como máximo vale", err_bound_vc(Ein_pla, np.shape(x)[0], 3, delta))
print("Cota usando Etest (Hoeffding): ", err_bound_hoeffding(Etest_pla, np.shape(x_test)[0], 1, delta))

print ("PLA-Pocket: Eout como máximo vale, con una probabilidad de ", str(1 - delta), ":")
print("Cota usando Ein (dimensión VC): Eout como máximo vale", err_bound_vc(Ein_pocket, np.shape(x)[0], 3, delta))
print("Cota usando Etest (Hoeffding): ", err_bound_hoeffding(Etest_pocket, np.shape(x_test)[0], 1, delta))

print ("RL: Eout como máximo vale, con una probabilidad de ", str(1 - delta), ":")
print("Cota usando Ein (dimensión VC): Eout como máximo vale", err_bound_vc(Ein_RL, np.shape(x)[0], 3, delta))
print("Cota usando Etest (Hoeffding): ", err_bound_hoeffding(Etest_RL, np.shape(x_test)[0], 1, delta))

