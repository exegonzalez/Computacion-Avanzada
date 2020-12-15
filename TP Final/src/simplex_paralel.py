#!/usr/bin/env python3

import numpy as np
import logging
import time
from mpi4py import MPI
import sys

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger()

def matrix_product(A,B):
    
  #Controla las dimensiones de las matrices recibidas como parametro
  #expresada en forma de tupla (n, m) n: filas; m: columnas
  if (len(A.shape) == 1): # Si es de 1D, es decir ( ,m)
    A = np.atleast_2d (A) # Devuelve (1, m)

  if (len(B.shape) == 1): # Si es de 1D, es decir (n, )
    B = B.reshape(-1,1) # Devuelve (n, 1)

  C = np.empty((A.shape[0],B.shape[1])) #Matriz de coeficientes tecnologicos
  rows_A = A.shape[0]
  cols_B = B.shape[1]
  cols_A = A.shape[1]
              
  #Se calcula la cantidad de columnas que le corresponde a cada proceso
  cols_per_proces = cols_A // (total_processes)

  #Para el caso en el que los datos a repartir y los procesos no son múltiplos
  rest = cols_A % (total_processes)
  if (rank == (total_processes-1)) and rest != 0:
    cols_per_proces += rest

  for i in range (rows_A):
    for j in range (cols_B):

      suma = 0

      """
      ****************************************************************************************
      ##################### START MULTIPLICATION OF MATRICES IN PARALLEL #####################
      ****************************************************************************************
      """
      
      if rank == 0:

        #Se reparten las columnas de A que van a trabajar los procesos
        process = 1
        for col in range(cols_per_proces,cols_A,cols_per_proces):

          if (process == (total_processes)) and (cols_A % (total_processes) != 0):
            break
          else:
            comm.send(col,dest=process)
          process += 1

        #El proceso 0 hace su parte
        for k in range(0,cols_per_proces):
          suma += A[i,k]*B[k,j]

        #El proceso 0 recibe los resultados y acumula todo en la matriz resultado
        for proc in range(1,total_processes):
          suma += comm.recv(None,source=proc)

        C[i,j] = suma
      
      else:

        #Cada proceso recibe su parte y resuelve el calculo
        col = comm.recv(None,source=0)
        for k in range(col,col+cols_per_proces):
          suma += A[i,k]*B[k,j]
        
        comm.send(suma,dest=0)

      """
      ****************************************************************************************
      ###################### END MULTIPLICATION OF MATRICES IN PARALLEL ######################
      ****************************************************************************************
      """

  #Se envia una copia del resultado de la multiplicacion a todos los procesos
  C = comm.bcast(C,0)

  #Se pasa a 1 dimension si la matriz resultante es una sola columna o fila
  if (C.shape[0] == 1 or C.shape[1] == 1):
    C = C.flatten()

  return C

def simplex_init(c, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], 
                         equalities=[], eqThreshold=[], maximization=True, M=1000.):
  '''
    Construye la primera base y los parámetros extendidos A' y c'. El orden en 
    el que quedan las restricciones es
    1) <=
    2) >=
    3) =

    En el mismo orden se agregan las variables 
    1) slack de <=
    2) exceso y artificial de >=
    3) artificial de =

    Parameters:
    c: Vector de coeficientes del funcional de las variables de decisión
    greaterThans: lista de listas con las filas de las restricciones de mayor o igual
    gtThreshold: lista de los lados derechos de las restricciones de mayor o igual
    lessThans: lista de listas con las filas de las restricciones de menor o igual
    ltThreshold: lista de los lados derechos de las restricciones de menor o igual
    equalities: lista de listas con las filas de las restricciones de igual
    eqThreshold: lista de los lados derechos de las restricciones de igual
    maximization: true si el problema es de maximización, false si el problema es de minimización
    M: penalización que se le asignará a las variables artificiales en caso de ser necesarias

    Returns:
    base: lista de indices correspondientes a las variables básicas de la base inicial
    c_p: vector c extendido con las variables slack, de exceso y artificiales
    A: matriz extendida
    b: vector de lados derechos de las restricciones
  '''

  # Inicialización
  amount_gt = len(gtThreshold)
  amount_lt = len(ltThreshold)
  amount_eq = len(eqThreshold)
  
  m = amount_gt + amount_lt + amount_eq
  n = len(c)
  n_p = m + n + amount_gt+ amount_eq # contando además las variables artificiales
  c_p = np.zeros(n_p)
  c_p[:n] = c if maximization else -c
  
  base = []
  
  A = np.empty((m, n_p))
  b = np.empty(m)
  if amount_lt > 0:
    A[0:amount_lt,:n] = lessThans
    b[0:amount_lt] = ltThreshold
  if amount_gt > 0:
    A[amount_lt:(amount_lt+amount_gt),:n] = greaterThans
    b[amount_lt:(amount_lt+amount_gt)] = gtThreshold
  if amount_eq > 0:
    A[(amount_lt+amount_gt):(amount_lt+amount_gt+amount_eq),:n] = equalities
    b[(amount_lt+amount_gt):(amount_lt+amount_gt+amount_eq)] = eqThreshold

  #Se calcula la cantidad de filas que le corresponde a cada proceso
  amount_per_process = amount_lt // (total_processes)
  rest = amount_lt % (total_processes)

  #Para el caso en el que las filas a repartir y los procesos no son múltiplos
  if (rank == (total_processes-1)) and rest != 0:
    amount_per_process += rest

  """
  ****************************************************************************************
  #################### START FIRST FOR OF MATRIX EXTENDED IN PARALLEL ####################
  ****************************************************************************************
  """

  if rank == 0:

    #Se envia a cada proceso su parte para calcular
    process = 1
    for i in range(amount_per_process,amount_lt,amount_per_process):
      if (process == (total_processes)) and (amount_lt % (total_processes) != 0):
        break
      else:                  
        comm.send(i,dest=process)
      process += 1

    for x in range(0,amount_per_process):
      A[:,n+x] = [(1. if x == j else 0) for j in range(m)]
      base.append(n+x)
    
    #Agrupa los vectores soluciones en la base y la matriz A
    for i in range(1,total_processes):
      result = comm.recv(None,source=i)
      for tupla in result:
        pos = tupla[1]
        A[:,n+pos] = tupla[0]
        base.append(n+pos)

    #En estos for, la cantidad de iteraciones son menor que la cantidad de procesos en los que se
    #puede repartir, por lo tanto no los paralelizamos y dejamos que el proceso 0 realice los calculos
    for i in range(amount_lt, amount_lt + 2*amount_gt, 2):
      A[:, n+i] = [(-1. if i == j else 0) for j in range(m)]
      A[:, n+i+1] = [(1. if i == j else 0) for j in range(m)]
      c_p[n+i+1] = -M
      base.append(n+i+1)
  
    for i in range(amount_lt + 2*amount_gt, amount_lt + 2*amount_gt + amount_eq):
      A[:, n+i] = [(1. if i == j else 0) for j in range(m)]
      c_p[n+i] = -M
      base.append(n+i)

  else:
    
    #Cada proceso recibe y guarda en una lista las tuplas con el vector y la posición correspondiente en cada una
    i = comm.recv(None,source=0)
    result = []
    for x in range(i,i+amount_per_process):
      result.append(([(1. if x == j else 0) for j in range(m)],x))  
    comm.send(result,dest=0)  

  """
  ****************************************************************************************
  ##################### END FIRST FOR OF MATRIX EXTENDED IN PARALLEL #####################
  ****************************************************************************************
  """

  #Se envia una copia de la inicializacion a todos los procesos
  base = comm.bcast(base)
  c_p = comm.bcast(c_p)
  A = comm.bcast(A)
  b = comm.bcast(b)

  return base, c_p, A, b


def solve_linear_program(base, c_p, A, b):
  '''
  Resuelve el programa lineal dado por los parámetros c_p, A y b, partiendo de 
  la base inicial "base". La primer matriz básica resultante debe ser la matriz 
  identidad, lo cual se asume como precondición.

  Parameters:
  base: base inicial (lista de índices)
  c_p: vector de coeficientes del funcional
  A: matriz de coeficientes tecnológicos
  b: vector de lados derechos (se asumen todos sus elementos >= 0)

  Returns:
  x_opt: diccionario de variables básicas y sus respectivos valores
  z_opt: valor óptimo de la función objetivo
  B_1: matriz inversa de la base óptima (sirve para construir la tabla óptima 
  y hallar los precios sombra y costos reducidos, de ser necesario)
  '''

  #B = A[:, base]
  m = len(b)
  B_1 = np.eye(m)

  # Iteración del método
  zj_cj = np.round(matrix_product(c_p[base],matrix_product(B_1,A)) - c_p, 10)
  
  # esto de antes se podria mejorar calculando sólo para las VNB
  # Mientras pueda mejorar
  while any(x < 0 for x in zj_cj):

    # Determinar Variable Entrante
    ve = np.argmin(zj_cj)
    log.info("Nueva variable entrante: x_{}".format(ve))
    
    # Vector correspondiente a la variable entrante
    A_ve = matrix_product(B_1,A[:,ve])

    b_p = matrix_product(B_1,b)

    #Se calcula la cantidad de filas que le corresponde a cada proceso
    amount_per_process = m // (total_processes)
    rest = m % (total_processes)

    #Para el caso en el que las filas a repartir y los procesos no son múltiplos
    if (rank == (total_processes-1)) and rest != 0:
      amount_per_process += rest

    titas = []

    """
    ****************************************************************************************
    ######################## START CALCULATION OF TITAS IN PARALLEL ########################
    ****************************************************************************************
    """

    if rank == 0:

      #Se envia a cada proceso su parte para calcular las titas
      process = 1
      for i in range(amount_per_process,m,amount_per_process):

        if (process == (total_processes)) and (m % (total_processes) != 0):
            break
        else:
            comm.send(i,dest=process)
        process += 1

      #El proceso 0 hace su parte
      for x in range(0,amount_per_process):
        if A_ve[x] > 0:
          calculo = b_p[x]/A_ve[x]
        else:
          calculo = np.nan
        titas.append(calculo)

      #Agrupa los resultados en el vector de titas
      for i in range(1,total_processes):
        result = comm.recv(None,source=i)
        titas.extend(result)

    else:
    
      #Cada proceso calcula el valor de la tita en el caso que sea mayor a 0 y lo guarda en una lista
      i = comm.recv(None,source=0)
      result = []
      for x in range(i,i+amount_per_process):
        if A_ve[x] > 0:
          calculo = b_p[x]/A_ve[x]
        else:
          calculo = np.nan
        result.append(calculo)
      comm.send(result,dest=0)  

    """
    ****************************************************************************************
    ######################### END CALCULATION OF TITAS IN PARALLEL #########################
    ****************************************************************************************
    """

    comm.barrier()
    titas = comm.bcast (titas)


    if all(np.isnan(tita) for tita in titas):
      log.info("Problema no acotado")
      raise("Problema no acotado")

    # Determinar Variable Saliente
    vs = np.nanargmin(titas)
    log.info("Nueva variable saliente: x_{}".format(base[vs]))
    base[vs] = ve
    log.info("Nueva base: {}".format(base))

    # Actualizar matriz inversa B_1
    E = np.eye(m)
    E[:,vs] = A_ve
    E_1 = np.eye(m)

    E_1[:,vs] = [(-E[i, vs]/E[vs, vs] if i != vs else 1./E[vs, vs]) for i in range(m)]

    B_1 = matrix_product(E_1,B_1)

    zj_cj = np.round(matrix_product(c_p[base], matrix_product(B_1,A)) - c_p, 10)

  # Cuando ya no puede mejorar
  b_p = matrix_product(B_1,b)

  x_opt = {base[j]: b_p[j] for j in range(m)}
  
  z_opt = matrix_product(c_p[base], b_p)

  return x_opt, z_opt, B_1

np.random.seed(12345)
num_vars = 30
num_restrictions = 50
A = [np.random.rand(num_vars) for j in range(num_restrictions)]
c = np.random.rand(num_vars)
b = np.random.rand(num_restrictions)

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
total_processes = comm.Get_size()

start_time = time.time()

base, c_p, A, b = simplex_init(c, lessThans=A, ltThreshold=b, maximization=True, M=10.)
#base, c_p, A, b = simplex_init([300., 250., 450.], greaterThans=[[0., 250., 0.]], gtThreshold=[500.], lessThans=[[15., 20., 25.], [35., 60., 60.], [20., 30., 25.]], ltThreshold=[1200., 3000., 1500.], maximization=True, M=1000.)

x_opt, z_opt, _ = solve_linear_program(base, c_p, A, b)

tiempo = (time.time() - start_time)

if rank == 0:

  print("La solución es:")
  for j in x_opt:
    print("x_{} = {}".format(j, x_opt[j]))
    
  print("Esto produce un funcional de z = {}".format(z_opt))

  print("--- %s seconds ---" % tiempo)