# Práctico Final: Computación Avanzada

1. Estudie el código del algoritmo simplex revisado.
2. Identifique partes del algoritmo que puedan ser paralelizables y proponga alternativas para hacerlo.
3. Escriba una versión paralela del algoritmo simplex y compárela con la versión secuencial utilizando al menos 5 instancias de PL distintas. Recomendamos que las ejecuciones sean locales en la misma PC para hacer la comparación lo más justa posible.

## Simplex primal revisado(con método de la M)

Adaptado de [aquí](https://www.matem.unam.mx/~omar/math340/revised-simplex.html)

El algoritmo Simplex fue diseñado por George Dantzig en 1947. La versión que habitualmente utilizan los paquetes de software se denomina *simplex revisado*, la cual se describe a continuación.

### Forma matricial del algoritmo simplex

Partamos de la forma canónica de un problema de programación lineal de maximización. Sea ![](https://i.postimg.cc/BbcD6TGJ/image.png)el vector de las el vector de las *variables de decisión*. Sean además los parámetros ![](https://i.postimg.cc/pXsh6XWc/image.png)
La siguiente es la forma general de un modelo de programación lineal canónico de maximización. ![](https://i.postimg.cc/rwz0XXzV/image.png) sujeto a ![](https://i.postimg.cc/Pr5NYsWW/image.png)

#### Un ejemplo

![](https://i.postimg.cc/rmTwpvyt/image.png)

```bash
c = [3., 5.]
b = [4., 12., 18.]
A = [[1., 0.], [0., 2.], [3., 2.]]
```

### Forma estándar

El modelo canónico anterior puede ser llevado a la forma estándar para transformar el sistema de inecuaciones en un sistema de ecuaciones introduciendo variables de holgura no negativas. Para ello se agregan al vector X las mencionadas variables de holgura, produciendo ![](https://i.postimg.cc/QMYZ2w93/image.png).
También se producen cambios en los parámetros,
![](https://i.postimg.cc/2SzpVrnY/image.png)
De esta forma, el modelo en forma estándar queda ![](https://i.postimg.cc/yYjQp9Sq/image.png)sujeto a![](https://i.postimg.cc/ryCQzWjd/image.png).

El sistema de ecuaciones resultantes tiene  n+m  incógnitas para  m  ecuaciones, de modo que se debe fijar el valor de  n  variables para resolverlo.

Cuando los valores de las variables originales del problema se fijan en  0  la solución es evidente, ya que las variables slack (las que se introdujeron en el pasaje a la forma estándar) toman los valores de los lados derechos. Esta solución es la solución inicial de la que parte el algoritmo simplex cuando  b≥0.

```python
import numpy as np

n = len(c)
m = len(b)

# En esta inicialización se puede inferir la primera base
base = []

# Ampliamos el vector de coeficientes del funcional para reflejar la inclusión de las variables slack
c_p = np.zeros(n+m)
c_p[:n] = c

# También ampliamos la matriz de coeficientes tecnológicos
A_p = np.empty((m, m+n))
A_p[0:m,:n] = A

for i in range(m):
  A_p[:, n+i] = [(1. if i == j else 0) for j in range(m)]
  base.append(n+i)

print("El nuevo vector c' es {}".format(c_p))
print("La matriz ampliada es A' = \n{}".format(A_p))
```
```bash
El nuevo vector c es [3. 5. 0. 0. 0.]
La matriz ampliada es A = 
[[1. 0. 1. 0. 0.]
 [0. 2. 0. 1. 0.]
 [3. 2. 0. 0. 1.]]
```

### Estrategia del algoritmo simplex

Este algoritmo explora **soluciones básicas factibles**. Cada **solución básica** resulta de fijar en 0 el valor de  n  variables (llamadas en este contexto *variables no básicas*). Las  m  variables restantes forman una base (y por ello se denominan *variables básicas*). Llamemos B a la submatriz de A' que contiene sólo los vectores asociados a las variables básicas, y  X′B  al vector de variables básicas. El sistema de ecuaciones resultante es el siguiente:![](https://i.postimg.cc/85Fc9hCZ/image.png). Cuando B está formada por vectores linealmente independientes existe  B−1 , de modo que los valores de  X′B  pueden calcularse mediante ![](https://i.postimg.cc/x8DfZhhd/image.png). Puede apreciarse que en la solución inicial descripta más arriba,  B=B−1=I , la matriz identidad de dimensión  m×m , y que![](https://i.postimg.cc/3xRTt3NB/image.png).

Cuando los valores de las variables básicas son todos positivos se dice que la solución además es **factible**. Dos soluciones básicas se dicen **adyacentes** si su base asociada difiere en exactamente una variable. **El algoritmo simplex se mueve entre soluciones básicas factibles adyacentes hasta que encuentra la óptima**. En cada iteración se realice un intercambio de una variable no básica por una básica para formar una nueva base. Esta transformación se puede realizar de diversas formas (Gauss-Jordan, por caso), pero el simplex revisado utiliza una forma inteligente de actualización de la matriz inversa para no tener que recalcular toda la matriz A', ya que esta matriz inversa funciona como matriz cambio de base.

```python
B = A_p[:,base]
B_1 = B.copy()

X_B = np.matmul(B_1, b)

print("La solución inicial es:")

for j in range(m):
  print("x_{} = {}".format(base[j], X_B[j]))
```
```bash
La solución inicial es:
x_2 = 4.0
x_3 = 12.0
x_4 = 18.0
```

### Prueba de optimidad y selección de variable entrante

Para saber si una solución básica factible puede mejorar se utiliza la función objetivo. En primer lugar, dado que las variables no básicas se fijan en un valor de 0, son las variables básicas las que determinan el valor de la función objetivo para una base dada.

![](https://i.postimg.cc/CKTcz43q/image.png)

Para saber si la solución actual puede mejorar se puede utilizar la ecuación 0, es decir,

![](https://i.postimg.cc/hPXsswgm/image.png)

Si existe algún  j  para el cual  zj−cj<0 , entonces la solución puede mejorarse introduciendo la variable  x′j . De todos los negativos se elige el de mayor valor absoluto, y se selecciona el  j  mínimo en caso de empate. Tal variable deberá entrar en la base en la próxima iteración, y por tanto se la denomina **variable entrante**. Denotamos  x′VE  a tal variable.

```python
z = np.matmul(c_p[base], X_B)
print("El valor inicial del funcional es z = {} * {}^T = {}".format(c_p[base], X_B, z))


zj_cj = np.matmul(c_p[base], np.matmul(B_1, A_p)) - c_p

print("A continuación el criterio de optimidad")
for j in range(n+m):
  print("z_{0} - c_{0} = {1}".format(j, zj_cj[j]))

if any(x < 0 for x in zj_cj):
  print("Una nueva variable puede entrar")
  # Determinar Variable Entrante
  ve = np.argmin(zj_cj)
  print("Nueva variable entrante: x_{}".format(ve))
else:
  print("La base actual es la óptima")
```
```bash
El valor inicial del funcional es z = [0. 0. 0.] * [ 4. 12. 18.]^T = 0.0
A continuación el criterio de optimidad
z_0 - c_0 = -3.0
z_1 - c_1 = -5.0
z_2 - c_2 = 0.0
z_3 - c_3 = 0.0
z_4 - c_4 = 0.0
Una nueva variable puede entrar
Nueva variable entrante: x_1
```

### Selección de variable saliente

Para seleccionar la **variable saliente** se selecciona aquella variable básica que garantice que la solución permanezca factible. Para esto se calculan sendos cocientes  θi , de la siguiente manera

![](https://i.postimg.cc/mZJCLrfc/image.png)

Donde![](https://i.postimg.cc/TYxgSYzn/image.png) es cada uno de los componentes del vector columna resultante del producto ![](https://i.postimg.cc/VNKKsb8y/image.png). Si para algún  i , ![](https://i.postimg.cc/TYxgSYzn/image.png)≤0  ese cociente se descarta (en caso de descartarse todos los cocientes el problema no está acotado).

De estos cocientes se selecciona el mínimo, y la variable básica correspondiente sale de la base. Tal índice lo denotamos VS.

```python
# Vector correspondiente a la variable entrante para la base actual
A_ve = np.matmul(B_1, A_p[:, ve])

# Calculamos los cocientes tita
b_p = np.matmul(B_1, b)
titas = [(b_p[i]/A_ve[i] if A_ve[i] > 0 else np.nan) for i in range(m)]

print("Los cocientes correspondientes a la variable entrante son:")

for j in range(m):
  print("tita_{} = {}".format(j, titas[j]))

if all(np.isnan(tita) for tita in titas):
  print("Problema no acotado")
  raise("Problema no acotado")
# Determinar Variable Saliente
vs = np.nanargmin(titas)
print("Nueva variable saliente: x_{} (que está en le posición {} de la base)".format(base[vs], vs))
```
```bash
Los cocientes correspondientes a la variable entrante son:
tita_0 = nan
tita_1 = 6.0
tita_2 = 9.0
Nueva variable saliente: x_3 (que está en le posición 1 de la base)
```
### Cambio de base

Para realizar el cambio de base bien se podría calcular la matriz inversa de la nueva base, es decir, se reemplazaría el vector  A′VS  por  A′VE  en la matriz básica  B  y se calcularía  B−1 . Sin embargo este cálculo puede resultar muy costoso.

En su lugar puede usarse una reformulación ingeniosa del problema de hallar la matriz inversa de la nueva base. Teniendo en cuenta que la nueva base, B' puede escribirse como sigue:

B′=B⋅E 

donde  E=I , a excepción de la columna VS-ésima, que se reemplaza por ![](https://i.postimg.cc/DwJvP1x6/image.png). Puede demostrarse que
![](https://i.postimg.cc/fyqsxbX1/image.png)

B−1  es la matriz inversa de la base actual, y  E−1  se puede calcular muy fácilmente.  E−1  es exactamente igual a  E , pero su columna VS-ésima se calcula de la siguiente manera:

![](https://i.postimg.cc/wMZPqRDx/image.png)

De esta forma se puede actualizar fácilmente la matriz inversa para reflejar el cambio de base.

```python
print("Matriz A extendida: \n{}".format(A_p))
print("Base actual: {}\n Matriz Básica actual B = \n{}".format(base, B))
# Cambio de variables en la base
base[vs] = ve
print("Nueva base: {}".format(base))
# Actualizar matriz inversa B_1
E = np.eye(m)
E[:,vs] = A_ve

print("E = {}".format(E))
print("Nueva matriz básica B' = \n{}".format(np.matmul(B, E)))

E_1 = np.eye(m)
E_1[:,vs] = [(-E[i, vs]/E[vs, vs] if i != vs else 1./E[vs, vs]) for i in range(m)]
print("E^-1 = {}".format(E_1))

print("Matriz inversa de la nueva base:\n{}".format(np.matmul(E_1, B_1)))
```
```bash
Matriz A extendida: 
[[1. 0. 1. 0. 0.]
 [0. 2. 0. 1. 0.]
 [3. 2. 0. 0. 1.]]
Base actual: [2, 3, 4]
 Matriz Básica actual B = 
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Nueva base: [2, 1, 4]
E = [[1. 0. 0.]
 [0. 2. 0.]
 [0. 2. 1.]]
Nueva matriz básica B = 
[[1. 0. 0.]
 [0. 2. 0.]
 [0. 2. 1.]]
E^-1 = [[ 1.  -0.   0. ]
 [ 0.   0.5  0. ]
 [ 0.  -1.   1. ]]
Matriz inversa de la nueva base:
[[ 1.   0.   0. ]
 [ 0.   0.5  0. ]
 [ 0.  -1.   1. ]]
```
Este proceso se repite hasta que se cumpla la prueba de optimidad.

A continuación el código secuencial del algoritmo simplex. Además de lo descripto anteriormente se incorpora el manejo de problemas de minimización y restricciones de  ≥  y de  =  mediante el método de la M.

```python
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger()

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
  cant_gt = len(gtThreshold)
  cant_lt = len(ltThreshold)
  cant_eq = len(eqThreshold)
  
  m = cant_gt + cant_lt + cant_eq
  n = len(c)
  n_p = m + n + cant_gt+ cant_eq # contando además las variables artificiales
  c_p = np.zeros(n_p)
  c_p[:n] = c if maximization else -c
  
  base = []
  
  A = np.empty((m, n_p))
  b = np.empty(m)
  if cant_lt > 0:
    A[0:cant_lt,:n] = lessThans
    b[0:cant_lt] = ltThreshold
  if cant_gt > 0:
    A[cant_lt:(cant_lt+cant_gt),:n] = greaterThans
    b[cant_lt:(cant_lt+cant_gt)] = gtThreshold
  if cant_eq > 0:
    A[(cant_lt+cant_gt):(cant_lt+cant_gt+cant_eq),:n] = equations
    b[(cant_lt+cant_gt):(cant_lt+cant_gt+cant_eq)] = eqThreshold

  for i in range(cant_lt):
    A[:, n+i] = [(1. if i == j else 0) for j in range(m)]
    base.append(n+i)

  for i in range(cant_lt, cant_lt + 2*cant_gt, 2):
    A[:, n+i] = [(-1. if i == j else 0) for j in range(m)]
    A[:, n+i+1] = [(1. if i == j else 0) for j in range(m)]
    c_p[n+i+1] = -M
    base.append(n+i+1)

  for i in range(cant_lt + 2*cant_gt, cant_lt + 2*cant_gt + cant_eq):
    A[:, n+i] = [(1. if i == j else 0) for j in range(m)]
    c_p[n+i] = -M
    base.append(n+i)

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
  zj_cj = np.round(np.matmul(c_p[base], np.matmul(B_1, A)) - c_p, 10)
  # esto de antes se podria mejorar calculando sólo para las VNB

  # Mientras pueda mejorar
  while any(x < 0 for x in zj_cj):
    # Determinar Variable Entrante
    ve = np.argmin(zj_cj)
    log.info("Nueva variable entrante: x_{}".format(ve))
    # Vector correspondiente a la variable entrante
    A_ve = np.matmul(B_1, A[:, ve])
    # Calculamos los cocientes tita
    b_p = np.matmul(B_1, b)
    titas = [(b_p[i]/A_ve[i] if A_ve[i] > 0 else np.nan) for i in range(m)]
    
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
    
    B_1 = np.matmul(E_1, B_1)
    zj_cj = np.round(np.matmul(c_p[base], np.matmul(B_1, A)) - c_p, 10)
  # Cuando ya no puede mejorar
  b_p = np.matmul(B_1, b)
  x_opt = {base[j]: b_p[j] for j in range(m)}
  z_opt = np.matmul(c_p[base], b_p)
  return x_opt, z_opt, B_1
```

### Un PL de ejemplo

![](https://i.postimg.cc/Y9ngm1n7/image.png)

Recuerde que para las restricciones de mayor o igual se agrega además una variable artificial para constituir la primera base.

```python
base, c_p, A, b = simplex_init([300., 250., 450.], greaterThans=[[0., 250., 0.]], gtThreshold=[500.], lessThans=[[15., 20., 25.], [35., 60., 60.], [20., 30., 25.]], ltThreshold=[1200., 3000., 1500.], maximization=True, M=1000.)
x_opt, z_opt, _ = solve_linear_program(base, c_p, A, b)
print("La solución es:")
for j in x_opt:
  print("x_{} = {}".format(j, x_opt[j]))
print("Esto produce un funcional de z = {}".format(z_opt))
```
```bash
La solución es:
x_2 = 12.799999999999999
x_4 = 152.0
x_0 = 56.0
x_1 = 2.0
Esto produce un funcional de z = 23060.0
```
### Generación de instancias aleatorias para las pruebas

A continuación se presenta un código muy simple que sirve para generar instancias aleatorias de programación lineal. La primera línea fija la semilla del generador de números pseudoaleatorios para poder reproducir las instancias a los fines de las comparaciones.

```python
np.random.seed(12345)
num_variables = 3000
num_restricciones = 5000
A = [np.random.rand(num_variables) for j in range(num_restricciones)]
c = np.random.rand(num_variables)
b = np.random.rand(num_restricciones)

base, c_p, A, b = simplex_init(c, lessThans=A, ltThreshold=b, maximization=True, M=10.)
x_opt, z_opt, _ = solve_linear_program(base, c_p, A, b)
print("La solución es:")
for j in x_opt:
  print("x_{} = {}".format(j, x_opt[j]))
print("Esto produce un funcional de z = {}".format(z_opt))
```
```bash
Se truncaron las últimas líneas 5000 del resultado de transmisión.
x_3002 = 0.8931622525975068
x_3003 = 0.6157030826681741
x_3004 = 0.5317852682699116
x_3005 = 0.9159068457507542
x_3006 = 0.5942376328452887
.
.
.
x_7995 = 0.6044423001846728
x_7996 = 0.5147526000129806
x_7997 = 0.6123667739346804
x_7998 = 0.5934254164925409
x_7999 = 0.009659588850238722
Esto produce un funcional de z = 0.002379825447922298
```


