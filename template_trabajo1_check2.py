# -*- coding: utf-8 -*-
"""
Nombre Estudiante:  Jacobo Casado de Gracia
Práctica 1 de Aprendizaje Automático
Curso 2021-2022
"""
#%%

# Importación de las librerías necesarias para el desarrollo de la práctica

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sklearn.utils import shuffle

# Selección de la semilla para los procesos pseudoaleatorios

np.random.seed(1)

#%%

# Sección dedicada a funciones auxiliares.

def wait():
     """Introduce una espera hasta que se pulse el intro.
    Limpia el plot anterior."""

     input("(Pulsa [Enter] para continuar...)\n")
     plt.close()


'''
Esta función muestra una figura 3D con la función a optimizar junto con el 
óptimo encontrado y la ruta seguida durante la optimización. Esta función, al igual
que las otras incluidas en este documento, sirven solamente como referencia y
apoyo a los estudiantes. No es obligatorio emplearlas, y pueden ser modificadas
como se prefiera. 
    rng_val: rango de valores a muestrear en np.linspace()
    fun: función a optimizar y mostrar
    ws: conjunto de pesos (pares de valores [x,y] que va recorriendo el optimizador
                           en su búsqueda iterativa del óptimo)
    colormap: mapa de color empleado en la visualización
    title_fig: título superior de la figura
    
Ejemplo de uso: display_figure(2, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
'''
def display_figure(rng_val, fun, ws, colormap, title_fig):
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-rng_val, rng_val, 50)
    y = np.linspace(-rng_val, rng_val, 50)
    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y) 
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap=colormap, alpha=.6)
    if len(ws)>0:
        ws = np.asarray(ws)
        min_point = np.array([ws[-1,0],ws[-1,1]])
        min_point_ = min_point[:, np.newaxis]
        ax.plot(ws[:-1,0], ws[:-1,1], fun(ws[:-1,0], ws[:-1,1]), 'b', markersize=5)
        ax.plot(min_point_[0], min_point_[1], fun(min_point_[0], min_point_[1]), 'r*', markersize=10)
    if len(title_fig)>0:
        fig.suptitle(title_fig, fontsize=16)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    
    
   

#%%

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

#%%

""" Apartado 1: Implementación del algoritmo de gradiente descendente.
    Parámetros que recibe la función:
         - w_ini = punto inicial donde comienza el algoritmo.
         - lr: valor del learning rate usado en el algoritmo
         - fun = f(x, y): función a optimizar.
         - grad_fun = gradiente de la función a optimizar.
         - epsilon: mínimo valor admisible de f (criterio de parada opcional).
         - max_iters: número máximo de iteraciones (criterio de parada fijado por defecto a 50).
         
       El algoritmo devuelve:
         - Una lista con todos los puntos calculados a lo largo de la ejecución del algoritmo
         (para posteriormente graficarlos)
         - Una lista con los valores de la función en esos puntos
         - Valor del mínimo encontrado.
         - Número de iteraciones realizadas.
         
         También imprime, al final de la ejecución, información sobre
         la mejor iteración del algoritmo (donde se consigue un menor valor 
         de la función), y sobre la última iteración de éste
         ESTO ÚLTIMO SE UTILIZA EN EL APARTADO 3) DE ESTE EJERCICIO.         
"""
def gradient_descent(w_ini, lr, grad_fun, fun, epsilon = None, max_iters = 50):
  
    w = w_ini
    
    y = np.array([])
    
    # Almacenaremos en este array todos los puntos que va calculando el algoritmo.
    ws = np.array(w)
    
    iteraciones = 0
    stop = False
    
    # Guardaremos la mejor iteración, con su valor y el punto donde se consigue.
    evaluacion_anterior = fun(w[0], w[1])
    mejor_evaluacion = evaluacion_anterior
    mejor_punto = w
    
    
    
    # El algoritmo sigue iterando o bien cuando se alcanzan las iteraciones máximas o cuando Stop = False. 
    # Stop se modifica si se llega a al mínimo valor admisible de f (epsilon)
    while (stop == False and iteraciones < max_iters):

    # Actualizamos el punto usando el gradiente de la función en ese punto
      w = w - lr * grad_fun(w[0], w[1])
      
      # Añadimos el punto a la lista
      ws = np.append(ws, w, axis=0)
      
      iteraciones = iteraciones + 1
      
      # Evaluamos el punto en la nueva posición
      evaluacion = fun(w[0], w[1])
      
      # lo añadimos a la lista de valores de f
      y = np.append(y, evaluacion)
      
      if (evaluacion < mejor_evaluacion):
          numero_mejor_iteracion = iteraciones
          mejor_evaluacion = evaluacion
          mejor_punto = w

        # Comprobamos si se ha llegado al criterio de parada de mínimo valor, si es que
        # se ha especificado cota.
      if (epsilon != None):
        stop = (evaluacion < epsilon)
      else:
        stop = False
    
    # El proceso iterativo acaba. Se imprimen los valores y se devuelve 
    # lo obtenido.
    
    
    print ("Algoritmo GD en ", w_ini, "con learning rate de " + str(eta))
    print("   Valor en la iteración", iteraciones, ":", evaluacion, "\n   Alcanzada en el punto", w, "\n" )
    print ("   El menor valor de la función es obtenido en la iteración", numero_mejor_iteracion, "\n   Con un valor de",mejor_evaluacion, "\n   Alcanzada en el punto", mejor_punto )
    print("---")
    ws = np.reshape(ws, (int(len(ws)/2),2))
    
    return ws, w, iteraciones, y

#%%

""" Apartado 2. Minimización de la función E(u,v)
    Primero se declara tanto su función, como sus derivadas de primer orden 
    en e y v para hacer, finalmente, el gradiente.
         
"""

def E(u,v):
    return ( (u * v * np.exp(-u**2 - v**2))**2 )

# Derivada parcial de E con respecto a u
def dEu(u,v):
    return -4*u**3*v**2*np.exp(-2*u**2 - 2*v**2) + 2*u*v**2*np.exp(-2*u**2 - 2*v**2)
    
# Derivada parcial de E con respecto a v
def dEv(u,v):
    return -4*u**2*v**3*np.exp(-2*u**2 - 2*v**2) + 2*u**2*v*np.exp(-2*u**2 - 2*v**2)

# Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


""" Fijación de parámetros (punto inicial, iteraciones máximas, error mínimo,
                        learning rate).
         
"""
w_ini = np.array([0.5,-0.5])
maxIter = 10000000000
eta = 0.1
error2get = 1e-8

# Llamamos al algoritmo de gradiente descendente
ws, w, iteraciones, y = gradient_descent(w_ini, eta, gradE, E, error2get, maxIter)

""" Construcción de dos gráficas:
        1) Gráfica que muestra la evolución del valor de f conforme avanzan
        las iteraciones
        2) Gráfico 3D de f y la evolución del punto sobre la función.
        Esta función es proporcionada por los docentes.
         
"""
# Construcción del gráfico 1

x = np.linspace(start = 0., stop = iteraciones, num = iteraciones)
plt.plot(x, y, 'k')
plt.title('Valor de la función E conforme avanza el número de iteraciones\n learning rate = ' + str(eta) + '\n Punto inicial: ' + str(w_ini))
plt.xlabel("Iteraciones")
plt.ylabel("E(u,v)")
plt.show()

# Construcción del gráfico 2
display_figure(2, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente. \n learning rate = ' + str(eta) + '\n Punto inicial: ' + str(w_ini))

#%%

""" Apartado 3. Minimización de la función f(u,v)

    En este caso, para el apartado b) usaremos varios puntos de comienzo, 
    por lo que, fijando un punto en concreto, se calcula el algoritmo de 
    gradiente descendiente con ambos learning rate (0.1 y 0.01),
    cosa que también sirve para el apartado 1).
    El apartado 1) fija como punto inicial el punto (-1,1) 
    y el apartado B es con una lista de puntos;
    cambiar el punto inicial para ver los resultados en cada punto.
    
    Como se comentó al principio, en el algoritmo de GD creado por mí,
    hago que imprima al final de la ejecución, información sobre
    la mejor iteración del algoritmo (donde se consigue un menor valor 
    de la función), y sobre la última iteración de éste,
    información con la que he hecho las tablas de la memoria.
    
    Primero se declara tanto su función, como sus derivadas de primer orden 
    en x y y para hacer, finalmente, el gradiente y posteriormente aplicar
    el algoritmo de GD.

"""

# Función f del apartado 3.
def f(x,y):
    return  x**2 + 2*y**2 + 2 * np.sin(2 * np.pi*x) * np.sin(np.pi * y) 

# Derivada parcial de f con respecto a x
def dfx(x,y):
    return 2*x + 4 * np.pi * np.sin(np.pi * y) * np.cos(2 * np.pi * x)

# Derivada parcial de f con respecto a y
def dfy(x,y):
    return 4*y + 2 * np.pi * np.sin(2* np.pi * x) * np.cos(np.pi * y)

# Gradiente de la función f
def gradxy(x,y):
  return np.array([dfx(x,y), dfy(x,y)])

""" Fijación de parámetros (punto inicial, iteraciones máximas, error mínimo,
                        learning rate).
         
"""

# FIJACIÓN DEL PUNTO INICIAL: PUNTO QUE HAY QUE CAMBIAR PARA HACER
# LAS COMPARATIVAS.
# No he hecho un bucle for, para poder probar con puntos diferentes
# y no generalizar en un solo bucle; cambiar el punto en esta variable.
w_ini = np.array([-1.0, 1.0])

# Para este apartado fijamos las iteraciones a 50
maxIter = 50
# Para este apartado no fijamos un valor mínimo
error2get = None

# Fijamos el learning rate (se ha probado con 0.1 y 0.01)
# Se hace un for para hacer una gráfica dual donde se muestra 
# el avance de la función a lo largo de las iteraciones
# con ambos lr.

# PARA CADA learning rate:
for eta in [0.01, 0.1]:
    
    # Se ejecuta el algoritmo del GD con esta función, y los parámetros de arriba
    ws, w, iteraciones, y = gradient_descent(w_ini, eta, gradxy, f, None, maxIter)
    
    # Se añade al plot el avance de la función respecto a las iteraciones
    x = np.linspace (start = 0., stop = iteraciones, num = iteraciones)
    plt.plot(x, y, label = 'learning rate = ' + str(eta))
    
# Se dibujan ambos plots en el mismo marco, con la leyenda para saber de qué función es cada traza.
plt.title("Valor de la función f conforme avanza el número de iteraciones, \ncon ambas tasas de aprendizaje")
plt.xlabel("Iteraciones")
plt.ylabel("f(x,y)")
plt.legend()
plt.show()

    
# Se hace otro for para hacer un plot 3D del avance del punto en la función.
# No lo hago dentro del for anterior porque se colisionaban las gráficas 2D y 3D.
for eta in [0.01, 0.1]:
    ws, w, iteraciones, y = gradient_descent(w_ini, eta, gradxy, f, None, maxIter)
    display_figure(3, f, ws, 'hsv','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente. \n learning rate = ' + str(eta) + '\n Punto inicial: ' + str(w_ini))
    

#%%
###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

#%%

# Funcion para calcular el error
def Err(x,y,w):
    
    w_t = np.transpose(w)
    
    predictions = np.array([np.dot(w_t, x_n) for x_n in x])
    
    error = predictions - y
    
    error_cuadratico = error * error 
    
    ecm = np.mean(error_cuadratico)
    
    return ecm

def grad_Err(x,y,w):
    
    w_t = np.transpose(w)
    
    d_w0 = np.mean( ( np.array([np.dot(w_t, x_n) for x_n in x]) - y) * x[:,0]) * 2
    d_w1 = np.mean( ( np.array([np.dot(w_t, x_n) for x_n in x]) - y) * x[:,1]) * 2
    d_w2 = np.mean( ( np.array([np.dot(w_t, x_n) for x_n in x]) - y) * x[:,2]) * 2
    
    return np.array([d_w0, d_w1, d_w2])

def grad_Err(x,y,w):
    
    w_t = np.transpose(w) 
    
    array_dw = np.empty(0)
    
    for i in range(x.shape[1]):
        dw = np.mean( ( np.array([np.dot(w_t, x_n) for x_n in x]) - y) * x[:,i]) * 2
        #print(dw)
        array_dw = np.append(array_dw, dw)    
        
    return array_dw

def get_batches(x, y, i, batch_size):
    
    x_batch = x[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    
    return x_batch, y_batch
    
# Gradiente Descendente Estocastico
def sgd(batch_size, x, y, x_test, y_test, w, lr, maxErr, seed, maxIters):
    
    Ein = Err(x,y,w)
    
    iters = 0
    
    while (Ein > maxErr and iters < maxIters):
        iters = iters + 1
        x, y = shuffle(x, y, random_state=seed)
        
        for i in range(0, len(x), batch_size):
            
            x_batch_i, y_batch_i = get_batches(x,y,i, batch_size)
            
            w = w - lr * grad_Err(x_batch_i, y_batch_i, w)
            
    Ein = Err(x, y, w)
    Eout = Err(x_test, y_test, w)
    
    return w, Ein, Eout

# Pseudoinversa	
def pseudoinverse(X,Y):
    
    U,D,V_traspose = np.linalg.svd(X, full_matrices=False)
    D = np.diag(D)
    V = np.transpose(V_traspose)
    
    A = np.dot(V,np.linalg.inv(D))
    A = np.dot(A, np.transpose(U))
    
    w = np.dot(A, Y)
    
    return w

def hiperplano(w, x):
    # A es w2, B es w1, C es w0
    
    A = w[2]
    B = w[1]
    C = w[0]
    
    return ((-1*B*x)/A - C/A)

#%%

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy' , 'datos/y_train.npy' )
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

#%%

w = np.array([0.,0.,0.])

#print ('Bondad del resultado para grad. descendente estocastico:\n')
print ('Bondad antes de ejecutar :\n')

print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print("Gradiente:" ,grad_Err(x,y,w))

print ('Bondad después de ejecutar :\n')
w, Ein, Eout = sgd(32,x,y,x_test, y_test, w,0.01, 1e-8, 1, 200)
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

data_1 = np.array([x_i for x_i, y_i in zip(x,y) if y_i == -1])
data_5 = np.array([x_i for x_i, y_i in zip(x,y) if y_i == 1])

x1 = hiperplano(w, 0.)
x2 = hiperplano(w, 1.)

recta = ([x1, x2])

plt.scatter(data_1[:,1], data_1[:,2], c='blue')
plt.scatter(data_5[:,1], data_5[:,2], c='red')

plt.plot(recta)

plt.xlabel("Intensidad media")

plt.xlabel("Simetría")

plt.show()


#%%

w = pseudoinverse(x,y)

data_1 = np.array([x_i for x_i, y_i in zip(x,y) if y_i == -1])
data_5 = np.array([x_i for x_i, y_i in zip(x,y) if y_i == 1])

x1 = hiperplano(w, 0.)
x2 = hiperplano(w, 1.)

recta = ([x1, x2])

plt.scatter(data_1[:,1], data_1[:,2], c='blue')
plt.scatter(data_5[:,1], data_5[:,2], c='red')

plt.plot(recta)

plt.xlabel("Intensidad media")

plt.xlabel("Simetría")

plt.show()

print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
#%%

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

def apply_noise(y, n_points):
    changes = np.random.choice(n_points, int(0.1 * n_points), replace = False)
    y[changes] = -y[changes]
        

#%%

n_points = 1000

x = simula_unif(n_points, 2, 1)

x1 = x[:,0]
x2 = x[:,1]


plt.scatter(x1, x2, c = 'k', s=4)
plt.title('1000 puntos generados aleatoriamente entre los ejes del cuadrado [-1, 1]')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.show()



#%%

w = np.array([0.,0.,0.])

y = np.array([f(x1,x2) for x1, x2 in zip(x1,x2)])

apply_noise(y, n_points)

data_negative = np.array([x_i for x_i, y_i in zip(x,y) if y_i == -1])
data_positive = np.array([x_i for x_i, y_i in zip(x,y) if y_i == 1])

plt.scatter(data_negative[:,0], data_negative[:,1], c='blue', s = 5)
plt.scatter(data_positive[:,0], data_positive[:,1], c='red', s = 5)

plt.show()

# creamos el vector de características añadiendo 1, x1, x2
x = np.array([[1,x1,x2] for x1, x2 in zip(x1,x2)])

#%%
print ("Ein: ", Err(x,y,w))
# print ("Ein v2: ", Err2(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print ("Gradiente: ", grad_Err(x, y, w))

print ('Bondad después de ejecutar :\n')
w, Ein, Eout = sgd(32,x,y,x, y, w,0.01, 1e-8, 1, 200)
print ("Ein: ", Ein)
print ("Eout: ", Eout)


#%%

array_Ein = np.empty(0)
array_Eout = np.empty(0)


for i in range (1000):
    
    x_train = simula_unif(n_points, 2, 1)
    x1_train = x_train[:,0]
    x2_train = x_train[:,1]
    x_train = np.array([[1,x1_train,x2_train] for x1_train, x2_train in zip(x1_train,x2_train)])

    
    y_train = np.array([f(x1_train,x2_train) for x1_train, x2_train in zip(x1_train,x2_train)])
    apply_noise(y_train, n_points)
    
    x_test = simula_unif(n_points, 2, 1)
    x1_test = x_test[:,0]
    x2_test = x_test[:,1]
    x_test = np.array([[1,x1_test,x2_test] for x1_test,x2_test in zip(x1_test,x2_test)])
    
    y_test = np.array([f(x1_test,x2_test) for x1_test,x2_test in zip(x1_test,x2_test)])
    apply_noise(y_test, n_points)
    
    w = pseudoinverse(x_train, y_train)
    
    Ein = np.append(array_Ein, Err(x_train,y_train, w))
    Eout = np.append(array_Eout, Err(x_test, y_test, w))
    
Ein = np.mean(Ein)
Eout = np.mean(Eout)

print ("Ein: ",Ein)
print ("Eout: ", Eout)

#%%

array_Ein = np.empty(0)
array_Eout = np.empty(0)

for i in range (1000):
    
    w = np.array([0.,0.,0.])
    
    x_train = simula_unif(n_points, 2, 1)
    x1_train = x_train[:,0]
    x2_train = x_train[:,1]
    x_train = np.array([[1,x1_train,x2_train] for x1_train, x2_train in zip(x1_train,x2_train)])

    
    y_train = np.array([f(x1_train,x2_train) for x1_train, x2_train in zip(x1_train,x2_train)])
    apply_noise(y_train, n_points)
    
    x_test = simula_unif(n_points, 2, 1)
    x1_test = x_test[:,0]
    x2_test = x_test[:,1]
    x_test = np.array([[1,x1_test,x2_test] for x1_test,x2_test in zip(x1_test,x2_test)])
    
    y_test = np.array([f(x1_test,x2_test) for x1_test,x2_test in zip(x1_test,x2_test)])
    apply_noise(y_test, n_points)
    
    w, Ein, Eout = sgd(256,x_train,y_train,x_test,y_test, w, 0.01, 1e-8, 1, 20)
    
    Ein = np.append(array_Ein, Ein)
    Eout = np.append(array_Eout, Eout)
    
Ein = np.mean(Ein)
Eout = np.mean(Eout)

print ("Ein: ",Ein)
print ("Eout: ", Eout)



#%%

w = np.array([0.,0.,0.,0.,0.,0.])

x = simula_unif(n_points, 2, 1)
x1 = x[:,0]
x2 = x[:,1]

y = np.array([f(x1,x2) for x1, x2 in zip(x1,x2)])
apply_noise(y, n_points)

# creamos el vector de características añadiendo 1, x1, x2, x1x2, x1**2 y x2**2
x = np.array([[1,x1,x2, x1*x2, x1**2, x2**2] for x1, x2 in zip(x1,x2)])

w, Ein, Eout = sgd(32,x,y,x,y,w,0.01, 1e-8, 1, 200)
print ("Ein: ", Ein)
print ("Eout: ", Eout)

