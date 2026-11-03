# 🎓 Masterclass: Machine Learning desde los cimientos

Una guía para reaprender todo lo que hay en este curso, en el mismo orden en que lo usarás. Cada módulo te dice **qué es**, **por qué importa**, **el código mínimo** y **dónde está en el repo** para practicarlo.

> Cómo estudiar con esto: lee un módulo, abre el notebook de clase que se indica, ejecútalo celda a celda (`Shift + Enter`) y cambia valores para ver qué pasa. Después resuelve la actividad en `workspace/`.

---

## Índice

0. [El mapa: qué es Machine Learning](#0-el-mapa-qué-es-machine-learning)
1. [Las herramientas: terminal, Git, uv y Jupyter](#1-las-herramientas)
2. [NumPy: números en bloque](#2-numpy-números-en-bloque)
3. [pandas: tablas](#3-pandas-tablas)
4. [Visualización: matplotlib y seaborn](#4-visualización)
5. [Limpieza de datos y outliers](#5-limpieza-de-datos-y-outliers)
6. [Transformación, codificación, escalado y train/test](#6-transformación-codificación-escalado-y-traintest)
7. [Tu primer modelo: árboles de decisión](#7-tu-primer-modelo-árboles-de-decisión)
8. [Tu proyecto Titanic: dónde estás y qué sigue](#8-tu-proyecto-titanic)
9. [Glosario](#9-glosario)
10. [Plan de estudio](#10-plan-de-estudio)

---

## 0. El mapa: qué es Machine Learning

**Programación normal:** tú escribes las reglas. `si edad < 12 → niño`.
**Machine Learning:** le das **ejemplos con la respuesta** y el programa **descubre las reglas** solo.

```
Datos (ejemplos) + Respuestas  ──▶  ALGORITMO  ──▶  Modelo (reglas aprendidas)
Datos nuevos (sin respuesta)   ──▶  Modelo     ──▶  Predicción
```

### Vocabulario mínimo (apréndelo ya, sale en todo)

| Término | Qué es | En el Titanic |
| --- | --- | --- |
| **Fila / muestra** | Un ejemplo | Un pasajero |
| **Feature / variable / X** | Lo que el modelo **ve** para decidir | `sex`, `age`, `pclass`, `fare`... |
| **Target / objetivo / y** | Lo que el modelo **tiene que adivinar** | `survived` |
| **Entrenar (`fit`)** | El modelo mira X e y y aprende reglas | |
| **Predecir (`predict`)** | El modelo recibe solo X y da su y | |

> ⚠️ **Error muy común:** la variable objetivo **nunca** se usa para predecir; es lo que se predice. En tu proyecto escribiste *"usaríamos la columna Survived para predecir"*. La pregunta pedía lo contrario: *¿qué columna de X (sin contar `survived`) te dice más sobre `survived`?* Tu propio EDA da la respuesta: **`sex`** (las mujeres sobrevivieron muchísimo más).

### Dos tipos de problema (aprendizaje supervisado)

- **Clasificación:** y es una **categoría**. ¿Sobrevivió (0/1)? ¿Qué especie de flor? → Titanic, Iris, pingüinos, breast cancer.
- **Regresión:** y es un **número continuo**. ¿Cuánto cuesta la casa? → viviendas, California housing.

### El flujo completo de un proyecto (el curso sigue este orden)

```
1. Cargar datos         → pandas                (cap. 3)
2. Mirar (EDA)          → pandas + gráficas     (cap. 3)
3. Limpiar              → duplicados, faltantes, outliers   (cap. 3)
4. Preparar             → crear variables, codificar, escalar, separar train/test   (cap. 3)
5. Entrenar             → un modelo (árbol)     (cap. 5)
6. Evaluar              → métricas, validación cruzada      (cap. 5)
7. Mejorar              → ajustar hiperparámetros           (cap. 5)
```

El 80 % del trabajo real está en los pasos 2-4. Por eso el curso dedica tanto tiempo a ellos.

---

## 1. Las herramientas

### 1.1 La terminal (PowerShell)

Es una ventana donde escribes órdenes en lugar de hacer clic. En VS Code: `Ctrl + ñ` (o *Terminal → New Terminal*). Comandos que vas a usar:

| Comando | Qué hace |
| --- | --- |
| `pwd` | ¿En qué carpeta estoy? |
| `ls` | ¿Qué hay en esta carpeta? |
| `cd carpeta` / `cd ..` | Entrar a una carpeta / subir una |

Tu repo está en `C:\Users\Pc\Documents\ml-course\ml-course` (ojo: hay **dos** carpetas `ml-course` anidadas; la de dentro es la que tiene Git).

### 1.2 Git y GitHub (guardar tu historial)

Lee [docs/how-to-use-this-repo.md](../docs/how-to-use-this-repo.md). La idea:

```
Repo del profesor (riosinda) ──Sync fork──▶ Tu fork (OscarrGv) ◀──push / pull──▶ Tu PC
```

- **Regla de oro:** solo editas dentro de `workspace/`. Si editas `notebooks/`, al sincronizar habrá **conflictos**.
- Rutina al terminar: `git add workspace/` → `git commit -m "qué hice"` → `git push`.
- Rutina al empezar: *Sync fork* en GitHub → `git pull` → `uv sync`.
- Los CSV **no** se suben (están en `.gitignore`); se regeneran con el script de datos.

### 1.3 uv (el gestor del entorno)

Un **entorno** es una caja con una versión concreta de Python y de las librerías, solo para este proyecto. `uv` la crea en la carpeta `.venv/`.

| Comando | Qué hace |
| --- | --- |
| `uv sync` | Lee `pyproject.toml` e instala todo lo que el curso necesita |
| `uv run python scripts/download_data.py` | Ejecuta un script **dentro** del entorno (crea los CSV en `data/`) |
| `uv add paquete` | Añade una librería (nunca `pip install`) |

Librerías del curso (en [pyproject.toml](../pyproject.toml)): `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.

### 1.4 Jupyter Notebooks

Clase: [notebooks/lectures/chapter_00/00_intro_to_jupyter.ipynb](../notebooks/lectures/chapter_00/00_intro_to_jupyter.ipynb)

- Un notebook es una lista de **celdas**: de **código** (Python) o de **Markdown** (texto con formato).
- `Shift + Enter` ejecuta la celda. La **última línea** se muestra sola (no hace falta `print`).
- El **kernel** es el Python que está "vivo" detrás y **recuerda** las variables. Si ejecutas celdas desordenadas, el kernel puede tener valores que no ves en pantalla.
- Por eso la regla: **Restart & Run All** debe funcionar de arriba abajo. Hazlo antes de entregar.
- En VS Code, elige arriba a la derecha el kernel `.venv` (Python 3.12).

---

## 2. NumPy: números en bloque

Clase: [01_numpy.ipynb](../notebooks/lectures/chapter_03/01_numpy.ipynb) · Actividad: Parte 1 de [01_actividad_numpy_pandas](chapter_03/01_actividad_numpy_pandas.ipynb)

**Por qué:** las listas de Python operan elemento a elemento con bucles (lento). Un **array** de NumPy opera sobre todos a la vez (**vectorización**). Todo el ML por debajo son arrays.

```python
import numpy as np

precios = np.array([2.5, 3.0, 4.2])
precios * 2              # array([5. , 6. , 8.4])  ← sin bucle
precios.mean()           # 3.23

m = np.array([[1, 2, 3],
              [4, 5, 6]])
m.shape                  # (2, 3) → 2 filas, 3 columnas
m.ndim                   # 2 dimensiones
m[1, 2]                  # 6  → fila 1, columna 2 (se cuenta desde 0)
m[:, 0]                  # array([1, 4]) → todas las filas, columna 0
m.reshape(3, 2)          # mismos 6 números en forma 3x2
m.sum(axis=0)            # suma por columnas → [5, 7, 9]
m.sum(axis=1)            # suma por filas    → [6, 15]
```

### ⭐ Máscaras booleanas (la idea más importante del capítulo)

```python
ventas = np.array([120, 80, 200, 50])
ventas > 100             # array([ True, False,  True, False])  ← la máscara
ventas[ventas > 100]     # array([120, 200])                   ← filtrar con ella
```

Una comparación devuelve `True/False` por elemento; usarla entre corchetes **filtra**. pandas funciona exactamente igual.

---

## 3. pandas: tablas

Clase: [02_pandas.ipynb](../notebooks/lectures/chapter_03/02_pandas.ipynb) · Actividad: Parte 2 de la 01

- **`Series`** = una columna con etiquetas. **`DataFrame`** = una tabla (varias Series juntas).

### Explorar (lo primero que haces con cualquier dataset)

```python
import pandas as pd
df = pd.read_csv("data/titanic.csv")

df.head()                 # primeras 5 filas
df.shape                  # (filas, columnas)
df.info()                 # tipo de cada columna y cuántos no-nulos
df.describe()             # estadísticas de las numéricas
df.describe(include="object")   # resumen de las de texto
df.isna().sum()           # faltantes por columna
df.isna().mean() * 100    # % de faltantes  (mean de True/False = proporción de True)
df.duplicated().sum()     # filas repetidas
df["sex"].value_counts()  # cuántas veces aparece cada valor
df["embarked"].unique()   # valores distintos
```

### Seleccionar y filtrar

```python
df["age"]                          # una columna (Series)
df[["age", "fare"]]                # varias columnas (DataFrame) → doble corchete
df.loc[0, "age"]                   # loc = por ETIQUETA (nombre de fila/columna)
df.iloc[0, 3]                      # iloc = por POSICIÓN (número)
df[df["age"] < 12]                 # filtrar con máscara
df[(df["sex"] == "female") & (df["pclass"] == 3)]   # Y  → &   (cada condición entre paréntesis)
df[(df["pclass"] == 1) | (df["pclass"] == 2)]       # O  → |
df.sort_values("fare", ascending=False)
```

> ⚠️ En pandas no se usa `and` / `or`, sino `&` / `|`, y **cada condición va entre paréntesis**.

### Crear columnas, agrupar, combinar

```python
df["familia"] = df["sibsp"] + df["parch"] + 1        # columna nueva (vectorizada)
df.groupby("sex")["survived"].mean()                 # % supervivencia por sexo
df.groupby(["pclass", "sex"])["age"].median()        # mediana por grupo
pd.crosstab(df["embarked"], df["pclass"], normalize="index")   # tabla cruzada en %
pd.merge(tabla1, tabla2, on="id")                    # unir por columna común (como JOIN de SQL)
pd.concat([df1, df2])                                # apilar tablas
```

**`groupby` en una frase:** *separa* las filas por grupo → *aplica* una función a cada grupo → *combina* los resultados.

**`.transform("median")`** (lo usaste en el Titanic): igual que `groupby(...).median()`, pero devuelve un valor **por cada fila original**, así puedes usarlo para rellenar huecos fila a fila.

---

## 4. Visualización

Clase: [03_matplotlib.ipynb](../notebooks/lectures/chapter_03/03_matplotlib.ipynb) · Actividad: Parte 3 de la 01

- **matplotlib**: la base, control total. Forma recomendada: `fig, ax = plt.subplots()` y luego `ax.plot(...)`, `ax.set_title(...)`.
- **seaborn**: encima de matplotlib, gráficos estadísticos en una línea a partir de un DataFrame.

| Pregunta que te haces | Gráfico | Código |
| --- | --- | --- |
| ¿Cómo se reparte una variable numérica? | Histograma | `sns.histplot(data=df, x="age")` |
| ¿Hay valores extremos? ¿Comparo grupos? | Boxplot | `sns.boxplot(data=df, x="pclass", y="fare")` |
| ¿Cuántos hay de cada categoría? | Barras de conteo | `sns.countplot(data=df, x="sex")` |
| ¿Una media por categoría? | Barras | `sns.barplot(data=df, x="sex", y="survived")` |
| ¿Se relacionan dos numéricas? | Dispersión | `sns.scatterplot(data=df, x="age", y="fare")` |
| ¿Qué variables van juntas? | Mapa de calor | `sns.heatmap(df.corr(numeric_only=True), annot=True)` |

### Leer un boxplot

```
   o   ← outlier (fuera de los bigotes)
   ┬   ← bigote: hasta 1.5 × IQR
 ┌─┴─┐ ← Q3 (75 %)
 │ ─ │ ← mediana (50 %)
 └─┬─┘ ← Q1 (25 %)
   ┴
```

### Correlación (lo de tu notebook de breast cancer)

Número entre **-1 y 1** que mide si dos variables numéricas suben juntas en línea recta.
`1` = suben juntas · `-1` = una sube y otra baja · `0` = sin relación **lineal**.
⚠️ Correlación **no** es causa: en tu EDA viste que el puerto `C` "parecía" salvar gente, pero en realidad allí embarcaba más gente de 1.ª clase. La clase era la causa oculta (**variable de confusión**). ¡Buen hallazgo!

**Regla del curso:** toda gráfica va seguida de 1-2 frases con lo que observas.

---

## 5. Limpieza de datos y outliers

Clase: [04_limpieza_y_outliers.ipynb](../notebooks/lectures/chapter_03/04_limpieza_y_outliers.ipynb) · Actividad: [02_actividad_limpieza_outliers](chapter_03/02_actividad_limpieza_outliers.ipynb)

*"Basura entra, basura sale"*: un modelo solo es tan bueno como sus datos.

### 5.1 Diagnóstico → limpieza estructural

1. **Duplicados:** `df.duplicated().sum()`, `df.drop_duplicates()`. Pero piensa: ¿es la misma persona dos veces o dos personas iguales? (En el Titanic no hay nombre, así que **pueden ser personas distintas**.)
2. **Texto inconsistente:** `" Madrid"`, `"madrid"`, `"MADRID"` → `df["c"].str.strip().str.lower()`.
3. **Tipos:** números guardados como texto → `pd.to_numeric(...)`, fechas → `pd.to_datetime(...)`.

### 5.2 ⭐ Valores faltantes

Primero pregunta **por qué** faltan (¿al azar, o falta porque significa algo? En el Titanic, `deck` falta sobre todo en 3.ª clase: que falte **es información**).

| Estrategia | Cuándo |
| --- | --- |
| Eliminar filas (`dropna`) | Faltan muy pocas y al azar |
| Eliminar la columna | Faltan casi todas (ej. `deck` ≈ 77 %) |
| **Media** | Numérica **simétrica** |
| **Mediana** | Numérica **asimétrica** o con outliers (la media se deja arrastrar) |
| **Moda** | Categórica (ej. `embarked`) |
| Por grupo | Cuando el valor depende de otra columna (edad según clase y sexo) |
| Columna indicadora | Cuando "faltar" puede ser informativo (`tiene_deck` = 1/0) |

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")    # o "mean", "most_frequent"
```

### 5.3 ⭐ Outliers (valores atípicos)

**Un outlier no es automáticamente un error.** Un billete de 512 £ en 1.ª clase es raro pero **real**; una edad de 250 años es un **error**.

| Método | Regla | Ojo |
| --- | --- | --- |
| **Visual** | boxplot / histograma | Siempre empieza aquí |
| **Z-score** | \|valor − media\| / desviación > 3 | Supone forma de campana; **falla en variables asimétricas** como `fare` |
| **IQR** | fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR] | Robusto (usa cuartiles); en variables muy sesgadas marca muchos |
| **Criterio de negocio** | lo que es imposible en la realidad | El más importante |

Tratamientos: **no hacer nada** (si es real), **eliminar** (si es error), **winsorizar / capping** (recortar al límite con `.clip(lower, upper)`), o **transformar** con logaritmo.

---

## 6. Transformación, codificación, escalado y train/test

Clase: [05_transformacion_y_escalado.ipynb](../notebooks/lectures/chapter_03/05_transformacion_y_escalado.ipynb) · Actividades: [03](chapter_03/03_actividad_transformacion_escalado.ipynb) y [04](chapter_03/04_actividad_decisiones_preprocesamiento.ipynb)

**Un modelo solo entiende números**, sin huecos y (para muchos modelos) en escalas parecidas.

### 6.1 Transformar

- **Logaritmo** para variables muy asimétricas (cola larga a la derecha, como `fare` o ingresos): `np.log1p(x)` = `log(1 + x)`, que **no falla con ceros** (`log(0)` es −∞).
- **Feature engineering:** crear variables con sentido. `familia = sibsp + parch + 1`.
- **Binning:** convertir un número en tramos: `pd.cut(df["age"], bins=[0, 12, 18, 60, 100], labels=["niño", "adolescente", "adulto", "mayor"])`.

### 6.2 ⭐ Codificar categóricas

¿Por qué no poner `S=0, C=1, Q=2`? Porque el modelo creería que `Q > C > S` y que `Q − C = C − S`. Eso es falso para puertos.

| Tipo | Ejemplo | Codificación |
| --- | --- | --- |
| **Nominal** (sin orden) | puerto, sexo, isla | **One-hot**: una columna 0/1 por categoría |
| **Ordinal** (con orden) | "malo < regular < bueno" | **Ordinal**: 0, 1, 2 respetando el orden |
| Binaria | sí/no | Una sola columna 0/1 |

```python
pd.get_dummies(df, columns=["embarked"])                # one-hot rápido con pandas
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
OneHotEncoder(handle_unknown="ignore")                  # versión scikit-learn
OrdinalEncoder(categories=[["malo", "regular", "bueno"]])
```

`drop="first"` quita una columna del one-hot porque es redundante (si no es `C` ni `Q`, es `S`). Importa en modelos lineales; para árboles da igual.

### 6.3 ⭐ Escalar numéricas

Si `age` va de 0-80 y `fare` de 0-512, los modelos que miden **distancias** (KNN, redes, regresión con regularización) creen que `fare` pesa más. Escalar los pone en igualdad.

| Escalador | Fórmula | Resultado | Cuándo |
| --- | --- | --- | --- |
| `StandardScaler` | (x − media) / desviación | media 0, desviación 1 | Opción por defecto |
| `MinMaxScaler` | (x − min) / (max − min) | entre 0 y 1 | Imágenes, datos acotados; sensible a outliers |

Los **árboles de decisión no necesitan escalado** (solo preguntan "¿x < umbral?"; cambiar la escala no cambia el orden). Lo verás en el cap. 5.

### 6.4 ⭐⭐ Train / test y la fuga de información (lo más importante del capítulo)

Separas los datos en:

- **Train (80 %)**: con lo que el modelo **estudia**.
- **Test (20 %)**: el **examen final**, datos que el modelo nunca vio. Simula "pasajeros nuevos".

```python
from sklearn.model_selection import train_test_split
X = df.drop(columns="survived")
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

- `stratify=y`: mantiene el mismo % de supervivientes (≈38 %) en train y en test.
- `random_state=42`: la separación sale igual cada vez (reproducible).

**Fuga de información (*data leakage*)** = el modelo ve, de alguna forma, información del examen. Dos tipos:

1. **Columna que chiva la respuesta:** `alive` es `survived` escrito en texto. Con ella el modelo acierta 100 % en el notebook y 0 % de utilidad en la vida real (un pasajero nuevo no trae `alive`). ✅ Tú ya la eliminaste.
2. **Calcular estadísticas con test:** si calculas la mediana de `age`, la media del escalador o los límites del IQR **usando todos los datos**, el test ya "contaminó" tu preparación.

**El orden correcto:**

```
1. Quitar columnas que filtran la respuesta (alive...) y decidir duplicados   ← puede ir antes del split
2. SEPARAR train / test
3. fit (aprender medianas, medias, categorías) SOLO con train
4. transform en train y en test con lo aprendido de train
```

```python
escalador = StandardScaler()
X_train_esc = escalador.fit_transform(X_train)   # aprende media/desv de TRAIN y aplica
X_test_esc  = escalador.transform(X_test)        # solo aplica (¡nunca fit en test!)
```

### 6.5 🚀 `ColumnTransformer` + `Pipeline`: todo junto

Hacer cada paso a mano es fácil de equivocar y, cuando llega un pasajero nuevo, tendrías que repetirlo todo. Un **Pipeline** encadena los pasos y garantiza el orden correcto:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numericas   = ["age", "fare"]
categoricas = ["sex", "embarked"]

preprocesador = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("esc", StandardScaler())]), numericas),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
])

preprocesador.fit(X_train)            # aprende todo SOLO de train
X_test_listo = preprocesador.transform(X_test)
```

Patrón de scikit-learn que se repite en TODO: **`fit`** (aprender) → **`transform`** (aplicar) o **`predict`** (predecir).

---

## 7. Tu primer modelo: árboles de decisión

Clase: [chapter_05/01_arboles_de_decision.ipynb](../notebooks/lectures/chapter_05/01_arboles_de_decision.ipynb) (con el dataset Iris)

### 7.1 Cómo piensa un árbol

Es un juego de "20 preguntas". En cada paso elige la pregunta que **mejor separa** las clases:

```
                 ¿sex == female?
                /               \
             sí                  no
     ¿pclass <= 2?           ¿age <= 6.5?
      /       \               /        \
 sobrevive   ...          sobrevive   no sobrevive
```

"Mejor separa" se mide con la **impureza** (`gini` o `entropy`): una hoja es **pura** si todos sus ejemplos son de la misma clase.

### 7.2 Entrenar y predecir

```python
from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)
pred = modelo.predict(X_test)
```

### 7.3 ⭐ Overfitting vs. underfitting

- **Underfitting** (se queda corto): árbol muy pequeño, falla en train y en test. Como estudiar solo el índice del libro.
- **Overfitting** (memoriza): árbol enorme que acierta ~100 % en train pero falla en test. Como memorizar las respuestas de un examen viejo.
- Lo buscas: buen resultado en **test**, no en train.

Se controla con **hiperparámetros** (perillas que eliges tú antes de entrenar):

| Hiperparámetro | Qué limita |
| --- | --- |
| `max_depth` | Cuántas preguntas seguidas (profundidad) |
| `min_samples_split` | Mínimo de ejemplos para seguir dividiendo |
| `min_samples_leaf` | Mínimo de ejemplos en cada hoja |
| `criterion` | `"gini"` o `"entropy"` |

### 7.4 Validación cruzada (cross-validation)

Una sola separación train/test puede salir "con suerte". La validación cruzada parte **train** en 5 trozos (*folds*), entrena con 4 y valida con 1, cinco veces, y promedia:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy")
scores.mean(), scores.std()
```

### 7.5 Buscar los mejores hiperparámetros: `GridSearchCV`

Prueba todas las combinaciones de una "rejilla" usando validación cruzada y se queda con la mejor:

```python
from sklearn.model_selection import GridSearchCV
rejilla = {"max_depth": [2, 3, 4, 5, None], "min_samples_leaf": [1, 2, 4, 8]}
busqueda = GridSearchCV(DecisionTreeClassifier(random_state=42), rejilla, cv=cv)
busqueda.fit(X_train, y_train)
busqueda.best_params_
mejor_modelo = busqueda.best_estimator_
```

Si el modelo es un `Pipeline`, el nombre lleva el paso delante: `"arbol__max_depth"`.

**El test se usa una sola vez, al final.** Elegir hiperparámetros mirando el test es otra forma de fuga.

### 7.6 Métricas: ¿qué tan bueno es?

**Matriz de confusión** (clasificación binaria):

| | Predijo 0 | Predijo 1 |
| --- | --- | --- |
| **Real 0** | ✅ Verdadero negativo | ❌ Falso positivo |
| **Real 1** | ❌ Falso negativo | ✅ Verdadero positivo |

| Métrica | Pregunta que responde |
| --- | --- |
| **Accuracy** | ¿Qué % acerté en total? (engaña si las clases están desequilibradas) |
| **Precision** | De los que dije "sobrevive", ¿cuántos sobrevivieron? |
| **Recall** | De los que sobrevivieron, ¿a cuántos detecté? |
| **F1** | Equilibrio entre precision y recall |
| **AUC (ROC)** | ¿Qué tan bien ordena por probabilidad? 0.5 = azar, 1 = perfecto |

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(classification_report(y_test, pred))
```

**Línea base:** si dices "nadie sobrevive" siempre, aciertas ≈62 % en el Titanic. Tu modelo tiene que superar eso para valer algo.

**`feature_importances_`**: qué variables usó más el árbol. Comparálo con lo que dijo tu EDA.

---

## 8. Tu proyecto Titanic

Archivo: [chapter_03/proyecto_titanic.ipynb](chapter_03/proyecto_titanic.ipynb) · Enunciado: [projects/chapter_03/README.md](../projects/chapter_03/README.md)

### ✅ Lo que ya hiciste bien

- Detectaste la fuga de `alive` y las columnas redundantes (`class`, `embark_town`, `alone`, `who`, `adult_male`).
- `deck` → indicador `tiene_deck` + eliminar la columna. Muy buena decisión.
- `embarked` → moda; `age` → mediana por clase y sexo con `transform`.
- Comparaste IQR vs. Z-score y el antes/después.
- Descubriste la **variable de confusión** puerto ↔ clase. Eso es pensar como analista.

### 🔧 Cosas que revisar

1. **Pregunta de la fase 2:** la columna que elegirías para predecir no puede ser `survived` (es la respuesta). Cámbiala por la que tu EDA señala como más fuerte y justifícala con el % de supervivencia.
2. **Celdas ✍️ vacías** en la fase 3 (duplicados, faltantes, outliers, antes/después, pregunta clave): son el **35 % de la nota**. Para cada una escribe: problema → opciones → decisión → por qué → cuántas filas.
3. **Fuga sutil en `age`:** calculaste la mediana por grupo con **todo** el dataset, antes del split. La pregunta clave de la fase 4.4 va justo de eso. Tienes dos caminos (tú decides y lo justificas):
   - Dejarlo y explicar en la bitácora que es una fuga pequeña y por qué.
   - Hacerlo bien: mover la imputación de `age` **después** del split (con un `SimpleImputer` dentro del Pipeline, o calculando las medianas solo con `X_train`).
4. **Celda 77** usa `df` en vez de `df2`; no rompe nada, pero sé consistente.

### ➡️ La fase 4, paso a paso (pistas, no solución)

1. **Feature engineering:** `familia = sibsp + parch + 1`; tramos de edad con `pd.cut`. Valida cada una con `groupby(...)["survived"].mean()`.
2. **Log:** ¿qué variable tiene cola larga? (mira tu histograma). Usa `np.log1p` por los billetes de 0 £.
3. **Codificar:** `sex` → binaria; `embarked` → one-hot. Para `pclass`, piensa: ¿la distancia 1.ª→2.ª es igual a 2.ª→3.ª?
4. **Split:** 80/20, `stratify=y`, `random_state=42`. Comprueba `y_train.mean()` vs. `y_test.mean()`.
5. **Escalar:** `fit` solo con train.
6. **Bonus Pipeline:** usa el esqueleto de la sección 6.5.
7. **Verificación:** `X_train.isna().sum().sum() == 0`, todo numérico, mismas columnas en train y test, `survived` fuera de X, y los cuatro `shape`.

Después, en el **capítulo 5** usarás estos datos para entrenar un árbol: [actividad_titanic_arbol](../notebooks/exercises/chapter_05/01_actividad_titanic_arbol.ipynb).

---

## 9. Glosario

| Término | Significado corto |
| --- | --- |
| Array | Bloque de números de NumPy |
| DataFrame / Series | Tabla / columna de pandas |
| EDA | Análisis exploratorio: mirar los datos antes de tocarlos |
| Feature (X) | Variable de entrada |
| Target (y) | Lo que se predice |
| Imputar | Rellenar valores faltantes |
| Outlier | Valor muy alejado del resto (no siempre error) |
| IQR | Q3 − Q1, el rango del 50 % central |
| Z-score | Cuántas desviaciones está un valor de la media |
| Winsorizar / capping | Recortar valores extremos a un límite |
| One-hot | Una columna 0/1 por categoría |
| Escalar | Poner variables en rangos comparables |
| Train / Test | Datos para aprender / para el examen final |
| Estratificar | Mantener la proporción de clases al separar |
| Data leakage | El modelo ve información que no tendría en la vida real |
| `fit` / `transform` / `predict` | Aprender / aplicar / predecir |
| Pipeline | Cadena de pasos de preparación + modelo |
| Hiperparámetro | Ajuste que eliges tú antes de entrenar (`max_depth`) |
| Overfitting | Memorizar train y fallar en datos nuevos |
| Validación cruzada | Evaluar en varios trozos de train y promediar |
| Accuracy / Precision / Recall / F1 | Métricas de clasificación |

---

## 10. Plan de estudio

Una sesión por bloque. En cada una: lee el módulo → ejecuta la clase → haz/revisa la actividad → escribe 3 frases de lo que aprendiste.

| Sesión | Módulo | Practica con |
| --- | --- | --- |
| 1 | 0 y 1 | `chapter_00/00_intro_to_jupyter.ipynb` (haz sus ejercicios) |
| 2 | 2 | `chapter_03/01_numpy.ipynb` + Parte 1 de tu actividad 01 |
| 3 | 3 | `chapter_03/02_pandas.ipynb` + Parte 2 de tu actividad 01 |
| 4 | 4 | `chapter_03/03_matplotlib.ipynb` + Parte 3 de tu actividad 01 |
| 5 | 5 | `chapter_03/04_limpieza_y_outliers.ipynb` + tu actividad 02 |
| 6 | 6 | `chapter_03/05_transformacion_y_escalado.ipynb` + tus actividades 03 y 04 |
| 7 | 8 | **Termina el proyecto Titanic** (fase 3 escrita + fase 4) |
| 8 | 7 | `chapter_05/01_arboles_de_decision.ipynb` + actividad Titanic con árbol |

**Truco para reaprender:** antes de mirar una solución vieja tuya, intenta escribirla de cero. Si te atascas, pídele a Claude **una pista**, no la respuesta.
