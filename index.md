---
title: Presentación
short_title: Presentación
---

# Predicción de *default* en Lending Club

**Proyecto Integrador de Aprendizaje Automático** · Machine Learning · 2026-3

**Autores:** Samuel Chamorro · Jesús Barrios · Jacobo Londoño

---

## Objetivo

Construir un clasificador binario que prediga si un préstamo emitido por la plataforma
Lending Club terminará en **impago** (`default = 1`, *Charged Off*) o será **pagado en su
totalidad** (`default = 0`, *Fully Paid*); comparar el desempeño y el tiempo de cómputo del
modelo construido con **scikit-learn** frente al construido con **PySpark**; y aplicar
**LIME** para interpretar predicciones individuales.

## El conjunto de datos

| | |
|---|---|
| **Fuente** | [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle) |
| **Ventana temporal** | 2007 – 2018 Q4 |
| **Archivo de préstamos aceptados** | 2.260.701 filas × 151 columnas · 1,67 GB |
| **Archivo de préstamos rechazados** | 27.648.741 filas × 9 columnas · 1,78 GB |
| **Universo de modelado** | **1.345.310** préstamos con desenlace observado |
| **Tasa de impago** | 19,96 % (desbalance de 4,01 : 1) |

El conjunto se usa **completo, sin muestreo, submuestreo ni reducción de filas**, tal como
exige el diseño del estudio. El único muestreo que aparece en todo el trabajo está en cuatro
visualizaciones de densidad —donde es la practica habitual expresamente y donde dibujar 1,3
millones de puntos sería ilegible— y en cada caso está anotado dentro del propio gráfico,
mientras que el estadístico correspondiente se calcula siempre sobre la población completa.

## Estructura del libro

::::{grid} 1 1 2 2

:::{card} 1. Análisis Exploratorio de Datos
:link: 01_eda.ipynb

Estructura, calidad y relaciones iniciales de los datos. Análisis unidimensional y
bidimensional de 75 variables numéricas y 12 categóricas, estudio detallado de valores
faltantes y resumen ejecutivo con las decisiones que gobiernan el preprocesamiento.

**el análisis exploratorio del diseño**
:::

:::{card} 2. Preprocesamiento, Modelado y Comparación
:link: 02_modelado.ipynb

Preprocesamiento en los dos entornos, `RandomForestClassifier` con `GridSearchCV` y con
`ParamGridBuilder` + `CrossValidator`, métricas, interpretabilidad con LIME, comparación
medida de rendimiento y reflexión crítica.

**secciónadel diseño**
:::

::::

## Las tres decisiones que determinan el resultado

Este proyecto tiene una particularidad que conviene enunciar de entrada: **las decisiones que
más afectan a la validez del modelo no se toman durante el modelado, sino durante el análisis
exploratorio**. Tres en concreto.

### 1. El universo son 1.345.310 préstamos, no 2.260.701

La columna `loan_status` no tiene dos categorías, sino nueve. Además de `Fully Paid` y
`Charged Off`, contiene **878.317 préstamos `Current`** y 34.292 en mora o `Default` — es
decir, préstamos **todavía vivos, cuyo desenlace no existe aún**.

Aplicar literalmente la definición del objetivo sobre el archivo completo asignaría
`default = 0` a todos ellos. Eso no es una simplificación: es un error de etiquetado que
enseña al modelo que un préstamo joven y al día es evidencia de buen pago, cuando lo único que
indica es que **no ha tenido tiempo de fallar**. Y el sesgo no es aleatorio, porque se
concentra en las cohortes recientes y en los plazos de 60 meses, que son precisamente los de
mayor riesgo.

El universo se restringe por tanto a los préstamos **resueltos**, que suman exactamente
1.345.310 — coincidiendo con la magnitud declarada para el conjunto de trabajo.

### 2. Treinta y ocho columnas son fuga de información

De las 151 columnas, **38 se registran después del desembolso**: `recoveries`,
`total_pymnt`, `last_fico_range_*`, el bloque de `settlement_*` y el de *hardship*. Un modelo
que las use alcanza un AUC cercano a 1 y es completamente inservible, porque para un
solicitante nuevo esas columnas están vacías.

No es una sospecha: la sección 3.1 del EDA lo demuestra midiendo que su correlación con el objetivo
es un orden de magnitud superior a la de la mejor predictora legítima.

### 3. La señal es débil y está repartida

De las 75 variables numéricas analizadas, **una sola** alcanza un tamaño de efecto mediano
según el criterio de Cohen (`int_rate`, *d* = 0,67). Únicamente 9 superan *d* = 0,20, y la
**mediana de |d| en el conjunto es 0,0997**. Las distribuciones de las dos clases se solapan
casi por completo en todas las variables.

Esto tiene dos consecuencias. Justifica el uso de un ensamble de árboles, cuyo trabajo es
precisamente agregar decenas de señales débiles. Y fija una expectativa realista: **un AUC
alto habría sido motivo de sospecha, no de celebración** — señal de que se filtró alguna de las
38 columnas post-desembolso.

## Un hallazgo que merece destacarse

Las dos variables más predictivas del conjunto —`int_rate` y `grade`— **no describen al
solicitante: describen al prestamista**. `grade` es la calificación de riesgo que Lending Club
asignó, e `int_rate` es el precio que se deriva mecánicamente de ella; la asociación entre
ambas es prácticamente funcional (η² = 0,909).

Usarlas es legítimo y se conservan, pero significa que el modelo aprende en buena
medida a **replicar un modelo de riesgo preexistente** en lugar de construir uno propio. Es
una advertencia decisiva al leer las explicaciones de LIME: cuando LIME señala `int_rate` como
el factor dominante de una predicción, lo que muestra es que el modelo aprendió a confiar en la
valoración del prestamista.

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

## Reproducibilidad

**Entorno.** Python 3.11 con `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,
`scipy`, `statsmodels`, `pyspark` 4.2, `lime`, `missingno` y `pyarrow`. PySpark requiere un JDK
instalado y la variable `JAVA_HOME` definida; los notebooks la localizan automáticamente si el
JDK está en una ruta estándar.

**Orden de ejecución.** Los notebooks deben correrse en orden: el primero deja en `outputs/`
los artefactos (taxonomía de variables, plan de tratamiento de nulos, matrices de correlación)
que el segundo consume, de modo que no haya divergencia entre lo que el análisis concluyó y lo
que el modelo hace.

```bash
jupyter execute 01_eda.ipynb 02_modelado.ipynb
```

**Coste de cómputo.** El primer notebook tarda unos 5 minutos. El segundo, del orden de dos
horas, dominadas por la búsqueda de hiperparámetros de PySpark con la configuración de 400
particiones adoptada. Para validar que todo corre antes de comprometer ese
tiempo, el segundo notebook incluye un interruptor `MODO_PRUEBA` que ejecuta las 50 celdas con
una fracción de los datos.

**Datos.** Los dos CSV deben estar en `data_tarea_1/`, descomprimidos, con la estructura de
carpetas original de Kaggle. No se incluyen en el repositorio por tamaño (3,4 GB).

**Construcción de este libro.**

```bash
jupyter-book build --html
```

## Nota sobre el conjunto de préstamos rechazados

El proyecto incluye también `rejected_2007_to_2018Q4.csv`, con 27,6 millones de solicitudes
rechazadas. Queda fuera del alcance por una razón estructural y no de conveniencia: esas
solicitudes **nunca fueron préstamos**, así que no tienen ni pueden tener `loan_status`, y sin
variable objetivo no hay aprendizaje supervisado posible sobre ellas.

Su existencia sí importa para interpretar los resultados. Los préstamos aceptados no son una
muestra aleatoria de quienes solicitaron crédito, sino el resultado de un filtro de aprobación
que descartó a doce de cada trece solicitantes. El modelo estima por tanto la probabilidad de
impago **condicionada a haber sido aprobado**, y no puede extrapolarse a la población general
de solicitantes. Es la forma de sesgo de selección que la literatura de crédito conoce como
*reject inference*.
