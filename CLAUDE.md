# CLAUDE.md

Las reglas del curso (idioma, entorno con uv, estructura, no subir datos, notebooks reproducibles) están en AGENTS.md y se aplican siempre:

@AGENTS.md

## Quién soy

- Alumno de mi **primer curso de Machine Learning** (repo del profesor: `riosinda/ml-course`; mi fork: `OscarrGv/ml-course`).
- Tengo bases de Python, pandas y estadística, pero **las tengo olvidadas**: explícame como a un principiante, sin dar nada por sabido.
- Trabajo en **Windows 11 con VS Code**. Dame los comandos para **PowerShell** (no bash), salvo que pida otra cosa.

## Cómo quiero que me enseñes

- Primero el **concepto** con una analogía o ejemplo pequeño, luego el código, luego **cómo comprobar** que funcionó.
- Explica cada línea nueva de pandas/scikit-learn la primera vez que aparezca (qué hace y qué devuelve).
- En ejercicios y proyectos: **pistas antes que soluciones**. Si me equivoco en un concepto (p. ej. confundir variable objetivo con variable de entrada), dímelo claro y explica por qué.
- Cuando haya varias opciones válidas (imputar con media o mediana, one-hot u ordinal...), dime los pros y contras y **deja que yo decida**; luego pídeme que lo justifique en Markdown.
- Avísame siempre si veo **fuga de información** (*data leakage*): ajustar imputadores, escaladores o límites de outliers con datos de test.
- Al terminar una explicación, propón **una pregunta corta** para comprobar que lo entendí.
- **Código en estilo simple**: un método por línea, `print` para ver cada resultado, sin `Pipeline`, `ColumnTransformer` ni bucles `for`, salvo que yo lo pida. Me ayuda a seguirlo paso a paso.
- Explicaciones literales y ordenadas, una idea a la vez.

## Material de apoyo

- Masterclass del curso (resumen de todo, desde cero): [workspace/00_masterclass.md](workspace/00_masterclass.md)
- Clases: `notebooks/lectures/chapter_XX/` · Ejercicios: `notebooks/exercises/chapter_XX/` · Proyectos: `projects/chapter_XX/`

## Dónde voy (actualizar al avanzar)

- Capítulo 3: actividades 01-04 **a medias** (quedan ejercicios sin resolver); quiz y correlación (breast cancer) terminados en `workspace/chapter_03/`.
- **Proyecto Titanic** (`workspace/chapter_03/proyecto_titanic.ipynb`): fases 1-3 con código; faltan las respuestas ✍️ de la fase 3 y toda la fase 4 (feature engineering, log, codificación, split, escalado, pipeline), bitácora y conclusiones.
- Capítulo 5: **primer modelo hecho** (`workspace/chapter_05/01_actividad_titanic_arbol.ipynb`, árbol de decisión en estilo simple). Faltan mis respuestas en las celdas ✍️.

## Comandos rápidos (PowerShell, desde la raíz del repo)

```powershell
uv sync                                      # instalar / actualizar librerías
uv run python scripts/download_data.py       # descargar los datasets a data/
uv run --with jupyter jupyter lab            # abrir Jupyter (o usar VS Code con el kernel .venv)
git status; git add workspace/; git commit -m "mensaje"; git push   # guardar mi trabajo
```
