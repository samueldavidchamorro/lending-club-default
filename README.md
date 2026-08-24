# Predicción de *default* en Lending Club

**Proyecto Integrador de Aprendizaje Automático** · Machine Learning · 2026-3
**Autores:** Samuel Chamorro · Jesús Barrios · Jacobo Londoño

Clasificación supervisada del impago de préstamos de Lending Club (2007–2018 Q4), con
comparación medida de rendimiento entre **scikit-learn** y **PySpark**, e interpretabilidad
mediante **LIME**.

---

## Contenido

| Notebook | Contenido |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | Análisis exploratorio: 75 variables numéricas y 12 categóricas, análisis unidimensional y bidimensional, valores faltantes, resumen ejecutivo | el análisis exploratorio |
| [`02_modelado.ipynb`](02_modelado.ipynb) | Preprocesamiento en ambos entornos, RandomForest con búsqueda de hiperparámetros, métricas, LIME, comparación y reflexión crítica | sección–|

El proyecto se publica además como **Jupyter Book** navegable; la portada está en
[`index.md`](index.md).

## Datos

| | |
|---|---|
| Fuente | [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle) |
| Préstamos aceptados | 2.260.701 filas × 151 columnas · 1,67 GB |
| Préstamos rechazados | 27.648.741 filas × 9 columnas · 1,78 GB |
| **Universo de modelado** | **1.345.310** préstamos con desenlace observado |
| Tasa de impago | 19,96 % (desbalance 4,01 : 1) |

Los CSV **no están en el repositorio** por tamaño. Hay que descargarlos de Kaggle y
descomprimirlos manteniendo la estructura original:

```
data_tarea_1/
├── accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv
└── rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv
```

## Instalación

```bash
conda create -n lending python=3.11
conda activate lending
pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels pyarrow pyspark lime missingno jupyter jupyter-book
```

PySpark necesita un JDK instalado y `JAVA_HOME` definido. Los notebooks lo localizan
automáticamente si el JDK está en una ruta estándar de Windows; en otro caso hay que
exportarlo a mano.

### Dos ajustes de PySpark que fallan de forma poco evidente

**`PYSPARK_PYTHON` debe apuntar al intérprete del entorno.** Spark lanza procesos *Python
worker* para cualquier operación que atraviese Python (UDF, `createDataFrame` desde objetos
locales, RDD de Python). Si no se le indica cuál usar, toma el `python` del `PATH`, que rara vez
es el del entorno virtual activo; y si ese intérprete no tiene `pyspark` instalado, el worker no
consigue reconectar con el *driver* y el trabajo muere con un mensaje que no menciona la causa:

```
SparkException: Python worker failed to connect back
SocketTimeoutException: Timed out while waiting for the Python worker to connect back
```

`02_modelado.ipynb` lo resuelve fijando `PYSPARK_PYTHON = sys.executable` antes de importar
`pyspark`, de modo que el worker sea siempre el mismo intérprete que ejecuta el notebook.

**El kernel de Jupyter debe registrarse con ruta absoluta.** Si el `kernel.json` del entorno
invoca `python` a secas, Jupyter arrancará el intérprete que encuentre en el `PATH` y el kernel
morirá al iniciar. Se registra correctamente una sola vez con:

```bash
python -m ipykernel install --user --name <entorno> --display-name "Python 3 (<entorno>)"
```

## Ejecución

Los notebooks deben correrse **en orden**: el primero deja en `outputs/` los artefactos
(taxonomía de variables, plan de tratamiento de nulos, matrices de correlación) que el segundo
consume, de modo que las decisiones del análisis y las del modelo no puedan divergir.

```bash
jupyter execute 01_eda.ipynb 02_modelado.ipynb
```

**Coste.** El EDA tarda unos 5 minutos. El modelado, del orden de dos horas, dominadas por la
búsqueda de hiperparámetros de PySpark, condicionada por la configuración de particionado
adoptada. Para validar que todo corre antes de comprometer ese tiempo, `02_modelado.ipynb`
incluye un interruptor `MODO_PRUEBA` que ejecuta las 50 celdas con una fracción de los datos.

## Construcción del libro

```bash
jupyter-book build --html
```

El sitio queda en `_build/html/` (portada, `eda/`, `modelado/` y las 57 figuras extraídas a
`build/`). Para servirlo con recarga en caliente durante la edición:

```bash
jupyter-book start
```

### Publicar en GitHub Pages

El sitio referencia sus recursos con **rutas absolutas** (`/build/figura.png`), que funcionan
en la raíz de un dominio pero **se rompen** si GitHub Pages lo sirve bajo un subdirectorio
—que es el caso de `https://<usuario>.github.io/<repositorio>/`—. Hay que indicar el prefijo
al construir:

```bash
BASE_URL=/lending-club-default jupyter-book build --html
```

En Windows con PowerShell:

```powershell
$env:BASE_URL = "/lending-club-default"; jupyter-book build --html
```

Sin esa variable el sitio se ve sin estilos y sin ninguna figura.

## Resultados

### Los dos motores, misma rejilla y mismo ganador

| | scikit-learn | PySpark |
|---|---|---|
| Hiperparámetros elegidos | 100 árboles, prof. 15 | 100 árboles, prof. 15 |
| **ROC AUC** | **0,7179** | **0,7154** |
| Precisión (clase 1) | 0,6095 | 0,6245 |
| Recall (clase 1) | 0,0389 | 0,0374 |
| F1 (clase 1) | 0,0731 | 0,0706 |
| Exactitud | 0,8032 | 0,8032 |

Los dos convergen al mismo modelo con una diferencia de AUC de **0,0025**. La discrepancia se
estrecha conforme crece el modelo —0,017 con 10 árboles y profundidad 5, frente a 0,0007 con
100 y profundidad 15— porque Spark discretiza las variables en `maxBins = 32` intervalos y esa
aproximación se diluye al aumentar la capacidad.

### El coste: 13,3 veces

| Fase | scikit-learn | PySpark | Factor |
|---|---|---|---|
| Preprocesamiento | 32,7 s | 119,3 s | 3,6× |
| Entrenamiento + validación cruzada | **661,3 s** | **8.941,4 s** | **13,5×** |
| Predicción | 1,2 s | 198,0 s | 165× |
| **Total** | **11,6 min** | **154,3 min** | **13,3×** |

**La hipótesis del diseño no se cumple**, y por razones estructurales: la matriz de diseño
cabe en menos de 1 GB sobre una máquina de 25,5 GB, así que Spark paga todo el coste de la
computación distribuida —particionado, serialización, planificación, histogramas aproximados
en cada nivel de cada árbol— sin poder cobrar ninguno de sus beneficios.

### El desbalance domina las métricas

Con el umbral 0,5 el *recall* es del 3,9 %: el modelo casi nunca predice impago, porque las
distribuciones de las dos clases se solapan y la probabilidad estimada raramente supera 0,5
para un préstamo con un 20 % de riesgo *a priori*. No es un fallo del clasificador sino del
umbral — `class_weight="balanced"` lleva el *recall* al **63,9 %** y el F1 de 0,073 a **0,433**
con el mismo AUC. La exactitud del 80,3 % es, casi exactamente, la del clasificador trivial.

### Lo que costó cada decisión de configuración

| Condición | Efecto medido |
|---|---|
| `.persist(MEMORY_AND_DISK)` | Un ajuste pasa de **106,6 s a 56,7 s**: la caché lo acelera **1,9×** |
| `shuffle.partitions = 400` | Un ajuste cuesta **109,0 s** con 400, **81,0 s** con 200 y **61,1 s** con 48. Pero con 48 la rejilla completa **desborda la memoria**: los dos parámetros están acoplados |
| `executor.memory = 8g` | **Ningún efecto.** En `local[*]` no hay *executors*; todo corre en la JVM del *driver* |

### ¿A partir de qué volumen gana PySpark?

| Filas de entrenamiento | scikit-learn | PySpark | Factor |
|---|---|---|---|
| 50.000 | 0,43 s | 32,06 s | **74,5×** |
| 400.000 | 5,70 s | 40,04 s | **7,0×** |

El factor cae de 74,5 a 7,0 al multiplicar por ocho el volumen, pero **no cruza la paridad**.
La pregunta está mal planteada: el umbral relevante no es de filas sino de arquitectura —
cuando los datos dejan de caber en una máquina, o cuando hay más de una.

## Licencia

Trabajo académico. Los datos pertenecen a sus titulares originales y están sujetos a los
términos de la fuente en Kaggle.
