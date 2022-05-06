# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: 
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle



# Fijamos la semilla
seed = 1
np.random.seed(seed)

# Función que devuelve 1 si el elemento es positivo y -1 si es negativo
def sign(x):
	if x >= 0:
		return 1
	return -1

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

def hiperplano(w, x):
    # A es w2, B es w1, C es w0
    
    A = w[2]
    B = w[1]
    C = w[0]
    
    return ((-1*B*x)/A - C/A)


#%%


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
x = simula_unif(50, 2, [-50,50])
x1 = x[:,0]
x2 = x[:,1]


# Ploteamos los datos en el cuadrado
plt.scatter(x1, x2, c = 'k', s=4)
plt.title(str(50) + ' puntos generados aleatoriamente entre los ejes del cuadrado [-50, 50] \n usando la función simula_unif')

# Damos título a los ejes del gráfico y mostramos la leyenda
plt.xlabel("Valor en x de los puntos")
plt.ylabel("Valor en y de los puntos")
plt.show()


#%%


x = simula_gauss(50, 2, [5,7])
x1 = x[:,0]
x2 = x[:,1]
# Ploteamos los datos en el cuadrado
plt.scatter(x1, x2, c = 'k', s=4)
plt.title(str(50) + ' puntos generados aleatoriamente entre los ejes del cuadrado [5, 7] \n usando la función simula_gauss')

# Damos título a los ejes del gráfico y mostramos la leyenda
plt.xlabel("Valor en x de los puntos")
plt.ylabel("Valor en y de los puntos")
plt.show()


#input("\n--- Pulsar tecla para continuar ---\n")


#%%


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

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

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def apply_noise(y, percentage):
    
    for label in [-1, 1]:
        # Seleccionamos los índices de las clases positivas (resp. negativas)
        idxs = np.where(y == label)[0]

        # Elegimos aleatoriamente una fracción 'p' de ellos
        random_idxs = np.random.choice(idxs, int(percentage * len(idxs)), replace = False)

        # Cambiamos el signo de los elegidos
        y[random_idxs] = -y[random_idxs]
        
        
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
    

def get_accuracy_recta(x, a,b, y):
    x1 = x[:,0]
    x2 = x[:,1]
    y_pred = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])
    
    ein = np.count_nonzero(y_pred-y)/np.shape(y)[0]
    print("Error de clasificación binario (tanto por uno de fallos): ", ein)
    

#%%
# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

intervalo = [-50,50]
x = simula_unif(100, 2, intervalo)
x1 = x[:,0]
x2 = x[:,1]
a,b = simula_recta(intervalo)
y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])

get_accuracy_recta(x, a,b,y)

# Graficamos los datos con sus etiquetas, incluyendo en la leyenda
# de qué clase son.
data_negative = np.array([x_i for x_i, y_i in zip(x,y) if y_i == -1])
data_positive = np.array([x_i for x_i, y_i in zip(x,y) if y_i == 1])
plt.scatter(data_negative[:,0], data_negative[:,1], c='red', label = 'Clase -1', s = 5)
plt.scatter(data_positive[:,0], data_positive[:,1], c='blue', label = 'Clase 1',  s = 5)

recta_x = np.linspace(-50,50,100)
recta_y = a*recta_x+b
formatted_a = '{0:.2f}'.format(a)
formatted_b = '{0:.2f}'.format(b)
plt.plot(recta_x, recta_y, '-k', label='f(x,y) = y - ax - b')

plt.xlim(intervalo)
plt.ylim(intervalo)
plt.title('100 puntos etiquetados por la recta f(x,y) = y - ax - b \n a = ' + str(formatted_a) + '; b = ' + str(formatted_b))
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend(loc = 1)
plt.show()

#%%

apply_noise(y, 0.1)
get_accuracy_recta(x, a,b,y)

# Graficamos los datos con sus etiquetas, incluyendo en la leyenda
# de qué clase son.
data_negative = np.array([x_i for x_i, y_i in zip(x,y) if y_i == -1])
data_positive = np.array([x_i for x_i, y_i in zip(x,y) if y_i == 1])
plt.scatter(data_negative[:,0], data_negative[:,1], c='red', label = 'Clase -1', s = 5)
plt.scatter(data_positive[:,0], data_positive[:,1], c='blue', label = 'Clase 1',  s = 5)

recta_x = np.linspace(-50,50,100)
recta_y = a*recta_x+b
formatted_a = '{0:.2f}'.format(a)
formatted_b = '{0:.2f}'.format(b)
plt.plot(recta_x, recta_y, '-k', label='f(x,y) = y - ax - b')

plt.xlim(intervalo)
plt.ylim(intervalo)
plt.title('100 puntos etiquetados por la recta f(x,y) = y - ax - b \n a = ' + str(formatted_a) + '; b = ' + str(formatted_b))
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend(loc = 1)
plt.show()

#%%

def f1(X):
    return (X[:,0]-10)**2 + (X[:,1]-20)**2 - 400

def f2(X):
    return 0.5 * (X[:,0]+10)**2 + (X[:,1]-20)**2 - 400

def f3(X):
    return 0.5 * (X[:,0]-10)**2 - (X[:,1]+20)**2 - 400

def f4(X):
    return X[:,1]-20*X[:,0]**2-5*X[:,0]+3

def etiquetar_puntos_f(f, x):
    y = np.sign(f(x))
    return y

def evaluar_f(f, x,y):

    y_hat = np.sign(f(x))
    #print(y_hat)
    #print(np.shape(y_hat))
    #print(np.shape(y))
    #errors = (y_hat - y)
    errors = (y_hat - y) != 0
    # print(errors)
    # errors = 0
    # hits = 0
    # for x, l in zip(y_hat, y):
    #     # Si está mal clasificado, actualizamos los pesos
    #     if sign(x) != l:
    #         errors = errors + 1
    #     else:
    #         hits = hits + 1
    # Hallamos la media de errores.
    Ein = np.mean(errors)
    print("Error de clasificación (Ein) obtenido: ", Ein)
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

# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')

#%%

y = etiquetar_puntos_f(f1, x)
apply_noise(y, 0.1)

print("Apartado 2.c.2 - Evaluación con f1")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)

# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')

#%%

y = etiquetar_puntos_f(f2, x)
apply_noise(y, 0.1)

print("Apartado 2.c.2 - Evaluación con f2")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)

# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')

#%%

y = etiquetar_puntos_f(f3, x)
apply_noise(y, 0.1)

print("Apartado 2.c.2 - Evaluación con f3")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)

# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')

#%%

y = etiquetar_puntos_f(f4, x)
apply_noise(y, 0.1)

print("Apartado 2.c.2 - Evaluación con f4")
print("f1:")
e1 = evaluar_f(f1, x, y)
print("f2:")
e2 = evaluar_f(f2, x, y)
print("f3:")
e3 = evaluar_f(f3, x, y)
print("f4:")
e4 = evaluar_f(f4, x, y)

# plot_datos_cuad(x, y, f1, title='Clasificación de los puntos con la función f1. \n. Error de clasificación (tanto por uno): ' + str(e1) , xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f2, title='Clasificación de los puntos con la función f2. \n. Error de clasificación (tanto por uno): ' + str(e2), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f3, title='Clasificación de los puntos con la función f3. \n. Error de clasificación (tanto por uno): ' + str(e3), xaxis='Eje X', yaxis='Eje Y')
# plot_datos_cuad(x, y, f4, title='Clasificación de los puntos con la función f4. \n. Error de clasificación (tanto por uno): ' + str(e4), xaxis='Eje X', yaxis='Eje Y')

#%%

def add_bias(x):
    x1 = x[:,0]
    x2 = x[:,1] 
    x_bias = np.array([[1,x1,x2] for x1, x2 in zip(x1,x2)])
    return (x_bias)

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
    
    accuracy = (hits / (errors + hits))
    #print(accuracy)
    
    return accuracy

def get_ein(w, x_bias, y):
    
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

# EJERCICIO 2.1: ALGORITMO PERCEPTRON
def ajusta_PLA(datos, label, max_iter, vini):
    
    datos_bias = add_bias(datos)
    
    w = vini.copy()  # No modificamos el parámetro w_ini
    evol = [vini]
    

    # Repetimos el algoritmo el número de iteraciones deseado
    for it in range(max_iter + 1):
        change = False

        # Recorremos todos los ejemplos
        for x, l in zip(datos_bias, y):
            # Si está mal clasificado, actualizamos los pesos
            if sign(x.dot(w)) != l:
                w += l * x
                change = True

        # Si no ha habido cambios, hemos terminado
        if not change:
            break

        # Guardamos la evolución de w
        evol.append(w.copy())
        
    print("-w final: ", w, "\n", "-Iteración final: ", it, "\n", "-Error de clasificación: ", get_ein(w, datos_bias, y))

    return w, it, evol, get_accuracy(w, datos_bias, y), get_ein(w, datos_bias, y)

def pla_pocket(datos, label, max_iter, vini):
    
    """Ajusta los parámetros de un hiperplano para un problema de clasificación
       binaria usando el algoritmo PLA-Pocket.
         - X: matriz de datos en coordenadas homogéneas (primera componente 1).
         - y: vector de etiquetas (1 ó -1).
         - max_it: número fijo de iteraciones.
         - w_ini: vector de pesos inicial."""
         
    datos_bias = add_bias(datos)
    w = vini.copy()
    w_best = w.copy()
    best_err = get_ein(w_best, datos_bias, label)
    evol = [vini]

    for _ in range(max_iter):
        for x, l in zip(datos_bias, label):
            if sign(x.dot(w)) != l:
                w += l * x

            curr_err = get_ein(w, datos_bias, label)
        
            if curr_err < best_err:
                best_err = curr_err
                w_best = w.copy()

        evol.append(w_best.copy())
        
    print("-w final: ", w_best, "\n", "-Iteración final: ", max_iter, "\n -Error de clasificación: ", get_ein(w, datos_bias, y))

    return w_best, evol, get_ein(w_best, datos_bias, y)
        

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
    
def plot_ein(evol, x, y):
    # Mostramos una gráfica con la evolución del accuracy
    acc_evol = []
    for w_ in evol:
        acc_evol.append(get_ein(w_, add_bias(x), y))
        
    plt.figure(figsize = (8, 6))
    plt.xlabel("Iteraciones")
    plt.ylabel("Accuracy")
    plt.title("Evolución del Ein (error de clasificación, tanto por uno) \n en la clasificación durante el algoritmo.")
    plt.plot(range(len(evol)), acc_evol)
    plt.show()
    
def plot_recta(x,y,intervalo,w):
    
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
    
    plt.xlim(intervalo)
    plt.ylim(intervalo)
    plt.title('Ajuste de la recta f(x,y) = y - ax - b sobre 100 puntos. \n a = ' + str(formatted_a) + '; b = ' + str(formatted_b))
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.legend(loc = 1)
    plt.show()
    
    
    

#%%

# Ejecutamos el algoritmo PLA con los datos del ejercicio 2.a 

# ES DECIR, SIN RUIDO.
intervalo = [-50,50]
x = simula_unif(100, 2, intervalo)
x1 = x[:,0]
x2 = x[:,1]
y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])

# Primero, lo ejecutamos con el [0,0,0]
print (" \n Ejecutando con w_ini = [0,0,0]. Resultados:")
w, it, evol, acc, ein = ajusta_PLA(x, y, 1000, [0.,0.,0.])

plot_recta(x,y,intervalo,w)
plot_accuracy(evol, x, y)
plot_ein(evol, x, y)

#%%
# Luego lo ejecutamos con vectores de números aleatorios en [0,1] 10 veces
# Y hacemos la media de las iteraciones.
# Imprimimos la información pedida en el enunciado.
iterations = []
error = []
for i in range(0,10):
    w_random =  np.random.rand(3)
    print("Ejecutando PLA con w inicial: ", w_random)
    w, it, evol, acc, ein  = ajusta_PLA(x, y, 1000, w_random)
    iterations.append(it)
    error.append(ein)
    
print (" \n Fin de las 10 ejecuciones. Resultados:")
print ("Iteración media después de las 10 ejecuciones: ", np.mean(iterations))
print ("Error medio de clasificación después de las 10 ejecuciones: ", np.mean(ein))


#%%

# Ahora, en el ejercicio 2.b, hacemos lo mismo pero añadimos ruido.
y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])
apply_noise(y, 0.1)
# Primero, lo ejecutamos con el [0,0,0]
print (" \n Ejecutando con w_ini = [0,0,0]. Resultados:")
w, it, evol, acc, ein = ajusta_PLA(x, y, 1000, [0.,0.,0.])

plot_recta(x,y,intervalo,w)
plot_accuracy(evol, x, y)
plot_ein(evol, x, y)

# Luego lo ejecutamos con vectores de números aleatorios en [0,1] 10 veces
# Y hacemos la media de las iteraciones.
# Imprimimos la información pedida en el enunciado.

iterations = []
error = []
for i in range(0,10):
    w_random =  np.random.rand(3)
    print("Ejecutando PLA con w inicial: ", w_random)
    w, it, evol, acc, ein  = ajusta_PLA(x, y, 1000, w_random)
    iterations.append(it)
    error.append(ein)

esqprint (" \n Fin de las 10 ejecuciones. Resultados:")
print ("Iteración media después de las 10 ejecuciones: ", np.mean(iterations))
print ("Error medio de clasificación después de las 10 ejecuciones: ", np.mean(ein))


#%%s

# Probamos a ajustar los datos con ruido con el PLA Pocket.

# Probamos con el [0,0,0] para hacer las gráficas
w, evol, ein = pla_pocket(x, y, 100, [0.,0.,0.])
plot_accuracy(evol, x, y)
plot_ein(evol, x, y)

# Luego probamos con 10 inicializaciones en aleatorio.
iterations = []
error = []
for i in range(0,10):
    w_random =  np.random.rand(3)
    print("Ejecutando PLA POCKET con w inicial: ", w_random)
    w, evol, ein = pla_pocket(x, y, 100, [0.,0.,0.])
    error.append(ein)
    
print ("Error medio de clasificación después de las 10 ejecuciones: ", np.mean(ein))


#%%
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def err(X, y, w):
    """Expresión del error cometido por un modelo de regresión logística.
         - X: matriz de características con primera componente 1.
         - y: vector de etiquetas.
         - w: vector de pesos."""

    return np.mean(np.log(1 + np.exp(-y * X.dot(w))))

def derr(x, y, w):
    """Expresión puntual del gradiente del error cometido por un modelo
       de regresión logística.
         - x: vector de características con primera componente 1.
         - y: vector de etiquetas.
         - w: vector de pesos."""

    return -y * x / (1 + np.exp(y * x.dot(w)))

def get_batches(x, y, i, batch_size):
    
    x_batch = x[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    
    return x_batch, y_batch



def logistic_sgd(x, y, lr, eps, batch_size):
    """Implementa el algoritmo de regresión logística con SGD. Devuelve el
       vector de pesos encontrado y las iteraciones realizadas.
         - X: matriz de datos, cada uno primera componente 1.
         - y: vector de etiquetas.
         - lr: valor del learning rate.
         - eps: tolerancia para el criterio de parada."""

    n = len(x)
    d = len(x[0])
    w = np.zeros(d)  # Punto inicial
    it = 0
    converged = False

    while not converged:
        w_old = w.copy()

        # Barajamos los índices y actualizamos los pesos
        x, y = shuffle(x, y, random_state=seed)
        for i in range(0, len(x), batch_size):
            
            x_batch_i, y_batch_i = get_batches(x,y,i, batch_size)
            
            idxs = np.arange(n)  # Vector de índices
            
            for idx in idxs:
                w -= lr * derr(x[idx], y[idx], w)

        # Comprobamos condición de parada
        converged = np.linalg.norm(w_old - w) < eps
        it += 1
        
    error = err(x,y,w)
        
    return w, it,error

def f(x, y, a, b):
	return signo(y - a*x - b)

#%%

w = [0,0,0]
intervalo = [0,2]
x = simula_unif(100, 2, intervalo)
x1 = x[:,0]
x2 = x[:,1]
a,b = simula_recta(intervalo)
y = np.array([f(x1,x2,a,b) for x1, x2 in zip(x1,x2)])

w, it, Ein = logistic_sgd(add_bias(x), y, 0.01, 0.01, 16)

print(w, it,Ein)

plot_recta(x,y,intervalo,w)

#%%


#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
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

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
