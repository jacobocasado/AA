# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: 
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sklearn.utils import shuffle

#%%

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

np.random.seed(1)

u, v = symbols('u v', real = True)
f = (u * v * exp(-u**2 - v**2))**2

#differntiating function f in respect to a
derivada_1_u = diff(f, u)
derivada_1_v = diff(f, v)

print (derivada_1_u)
print (derivada_1_v)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

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

def gradient_descent(w_ini, lr, grad_fun, fun, epsilon = None, max_iters = 50):
  
    w = w_ini
    ws = np.array(w)
    
    y = np.array([])

    # print ("w inicial: ", w)

    iteraciones = 0
    stop = False
    
    evaluacion_anterior = fun(w[0], w[1])
    mejor_evaluacion = evaluacion_anterior
    mejor_punto = w
    
    print("Evaluacion inicial, en el punto", w, ":", evaluacion_anterior)

    while (stop == False and iteraciones < max_iters):

      w = w - lr * grad_fun(w[0], w[1])
      
      ws = np.append(ws, w, axis=0)
      # print ("w modificado:", w)
      iteraciones = iteraciones + 1

      evaluacion = fun(w[0], w[1])
      y = np.append(y, evaluacion)
      
      if (evaluacion < mejor_evaluacion):
          numero_mejor_iteracion = iteraciones
          mejor_evaluacion = evaluacion
          mejor_punto = w

      # print("Evaluacion a ese valor de w: ", evaluacion)

      # print (evaluacion)

      if (epsilon != None):
        stop = (evaluacion < epsilon)
      else:
        stop = False
    
    
    print("Paro porque ", evaluacion, "es mayor a ", epsilon, "a la iteración", iteraciones, "y se alcanza en el punto",w )
    print ("La mejor iteracion es obtenida en ", numero_mejor_iteracion, "con un valor de ",mejor_evaluacion, "en el punto", mejor_punto )
    ws = np.reshape(ws, (int(len(ws)/2),2))
    
    return ws, w, iteraciones, y

# Ver que olos valores disminuyen


    
#%%

eta = 0.1
maxIter = 10000000000
error2get = 1e-8
w_ini = np.array([0.5,-0.5])

ws, w, iteraciones, y = gradient_descent(w_ini, eta, gradE, E, error2get, maxIter)

x = np.linspace ( start = 0.    # lower limit
                , stop = iteraciones      # upper limit
                , num = iteraciones      # generate 51 points between 0 and 3
                )

plt.plot(x, y, 'k')
plt.title("Valor de la función E conforme avanza el número de iteraciones")
plt.xlabel("Iteraciones")
plt.ylabel("E(u,v)")
plt.show()


display_figure(3, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')

#%%

x, y = symbols('x y', real = True)
f =  x**2 + 2*y**2 + 2 * sin(2 * pi*x) * sin(pi * y)

#differntiating function f in respect to a
derivada_1_x = diff(f, x)
derivada_1_y = diff(f, y)

print (derivada_1_x)
print (derivada_1_y)

# ejercicio2. 
def f(x,y):
    return  x**2 + 2*y**2 + 2 * np.sin(2 * np.pi*x) * np.sin(np.pi * y) 
print (f(-1.0,1.0))

def dfx(x,y):
    return 2*x + 4 * np.pi * np.sin(np.pi * y) * np.cos(2 * np.pi * x)


def dfy(x,y):
    return 4*y + 2 * np.pi * np.sin(2* np.pi * x) * np.cos(np.pi * y)

def gradxy(x,y):
  return np.array([dfx(x,y), dfy(x,y)])

eta = 0.01
maxIter = 50
error2get = None
w_ini = np.array([-2.0,2.0])

ws, w, iteraciones, y = gradient_descent(w_ini, eta, gradxy, f, None, maxIter)


x = np.linspace ( start = 0.    # lower limit
                , stop = iteraciones      # upper limit
                , num = iteraciones      # generate 51 points between 0 and 3
                )

plt.plot(x, y, 'k')
plt.title("Valor de la función f conforme avanza el número de iteraciones")
plt.xlabel("Iteraciones")
plt.ylabel("f(x,y)")
plt.show()

display_figure(3, f, ws, 'hsv','Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente')

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

def Err2(x,y,w):
    error = 0.0
    
    for i in range(x.shape[0]):
        error = error + (w.T.dot(x[i]) - y[i])**2
        
    return error/x.shape[0]


def grad_Err(x,y,w):
    
    w_t = np.transpose(w)
    
    d_w0 = np.mean( ( np.array([np.dot(w_t, x_n) for x_n in x]) - y) * x[:,0]) * 2
    d_w1 = np.mean( ( np.array([np.dot(w_t, x_n) for x_n in x]) - y) * x[:,1]) * 2
    d_w2 = np.mean( ( np.array([np.dot(w_t, x_n) for x_n in x]) - y) * x[:,2]) * 2
    
    return np.array([d_w0, d_w1, d_w2])


def get_batches(x, y, i, batch_size):
    
    x_batch = x[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    
    return x_batch, y_batch
    
# Gradiente Descendente Estocastico
def sgd(batch_size, x, y, x_test, y_test, w, lr, maxErr, seed, maxIters):
    
    Ein = Err(x,y,w)
    
    print(y.shape[0])
    iteraciones_batch = int(x.shape[0] / batch_size)
    iters = 0
    
    while (Ein > maxErr and iters < maxIters):
        iters = iters + 1
        # print(iters)
        x, y = shuffle(x, y, random_state=seed)
        
        for i in range(0, len(x), batch_size):
            
            x_batch_i, y_batch_i = get_batches(x,y,i, batch_size)
            
            w = w - lr * grad_Err(x_batch_i, y_batch_i, w)
            
            Ein = Err(x, y, w)
            Eout = Err(x_test, y_test, w)
            
    print ("Ein:", Ein)
    print ("Eout:", Eout)
    
    return w

# Pseudoinversa	
def pseudoinverse(X,y):
    
    U,D,V_traspose = np.linalg.svd(X, full_matrices=False)
    D = np.diag(D)
    V = np.transpose(V_traspose)
    
    A = np.dot(V,np.linalg.inv(D))
    A = np.dot(A, np.transpose(U))
    
    w = np.dot(A, y)
    
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
# print ("Ein v2: ", Err2(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print ("Gradiente: ", grad_Err(x, y, w))

print ('Bondad después de ejecutar :\n')
w = sgd(32,x,y,x_test, y_test, w,0.01, 1e-8, 1, 200)


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
print(w)

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
# print ("Ein v2: ", Err2(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
#%%

#Seguir haciendo el ejercicio...

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

#Seguir haciendo el ejercicio...

#%%

