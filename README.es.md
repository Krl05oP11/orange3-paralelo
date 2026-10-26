# Orange Paralelo

*[Read in English](https://github.com/Krl05oP11/orange3-paralelo/blob/main/README.md)*

Add-on para [Orange](https://orangedatamining.com/) 3.40 con dos cosas:
- 4 widgets que reparten el trabajo entre los núcleos del procesador;
- un modelo nuevo, **Linear SVM (liblinear)**.

Conviven con los originales en una categoría propia, **Paralelo**. La interfaz está en inglés:

| Widget | Qué se reparte |
|---|---|
| Test and Score (parallel) | Los pares (pliegue × modelo) de la evaluación, con todas las formas de muestreo del original. Suma la opción **«Keep groups together:»**: validación cruzada estratificada por la clase y agrupada por una variable (StratifiedGroupKFold), para datos con varias filas por grupo, como varios textos de una misma fuente |
| Parameter Fitter (parallel) | Los puntos de la grilla |
| Permutation Plot (parallel) | Las permutaciones |
| Random Forest (parallel) | Los árboles (`n_jobs` de scikit-learn) |
| Linear SVM (liblinear) | Nada: es un modelo nuevo, pensado para texto (detalle abajo). Tarda segundos por pliegue donde el SVM de Orange tarda minutos u horas |

En los 4 widgets paralelos, entradas, salidas y opciones son las del original. Cada uno suma una caja **Parallel** con el número de procesos: por defecto, los núcleos físicos menos uno. Con 1, corre el camino original de Orange, en serie.

**Linear SVM (liblinear):**
- **Modelos:** `LinearSVC` uno contra uno, como libsvm, en clasificación; `LinearSVR` en regresión.
- **Es determinista.**
- **Probabilidades calibradas** con una sigmoide sobre 5 pliegues internos. La clase más probable es siempre la predicha.
- **Con datos dispersos no imputa.** Escala cada columna por su rango, como las SVM de Orange, pero adentro del modelo, sin crear una variable por columna.

## Instalación
**Desde Orange:**
1. Abrí **Options → Add-ons… → Add more…**.
2. Escribí `orange3-paralelo`.
3. Hacé clic en **Add** y después en **OK**.
4. Reiniciá Orange.

**Con pip,** en el entorno de Python donde está Orange:
```bash
pip install orange3-paralelo
```

**Para desinstalarlo,** desmarcalo en el mismo diálogo o corré `pip uninstall orange3-paralelo`. Orange no se toca.

## Cuánto se gana
Medido en un Ryzen 9 7900 (12 núcleos, 24 hilos), con 11 procesos. En todos los casos los resultados fueron idénticos a los de Orange:

| Caso | Orange en serie | Paralelo (11 procesos) | Aceleración |
|---|---|---|---|
| Test and Score: corpus de texto, TF-IDF (2.699 × 70k) + regresión logística, CV 10 | 608 s | 140 s | **4,4×** |
| Test and Score: sintético 50k × 50 + Random Forest (100 árboles), CV 10 | 208 s | 25 s | **8,4×** |
| Parameter Fitter: sintético 10k × 30 + Random Forest, 10 puntos (10–100 árboles) | 101 s | 20 s | **5,2×** |
| Permutation Plot: heart_disease + regresión logística, 100 permutaciones | 8,2 s | 2,8 s | **3,0×** |
| Test and Score: titanic + regresión logística, CV 10 (corre en serie solo) | 0,0 s | 0,0 s | 1,0× |

- **Las tareas chicas corren en serie solas.** El umbral es 1 millón de celdas × modelos a ajustar (`motor.UMBRAL_TRABAJO`). Por debajo, la serie tarda menos de ~1 s, que es lo que cuesta arrancar los procesos.
- **La primera corrida en paralelo es ~1 s más lenta.** Pasa una vez por sesión de Orange y otra vez después de cancelar, porque se arrancan los procesos. Después se reutilizan.
- **En validación cruzada, el techo es el número de tareas.** 10 pliegues × 1 modelo son 10 tareas, y manda el pliegue más lento. Parameter Fitter y Permutation Plot tienen más tareas y aprovechan más núcleos.

### Con un corpus de texto real
El corpus tiene 2.699 textos de 18 clases, en 130.047 columnas de Bag of Words, armado como Corpus de Orange3-Text. Test and Score usó CV de 5, estratificada, y cada modelo, los ajustes de su widget:

| Caso | Antes | Con el add-on | Aceleración |
|---|---|---|---|
| Regresión logística (C = 1), 5 tareas | 7,8 min (Orange en serie) | 0,7 min | **11×**, con clases y probabilidades idénticas (1e-12) |
| Regresión logística + SVM de Orange, 10 tareas | ~113 min con una versión anterior del add-on; ~9 h estimadas con Orange en serie | **9,1 min** | **12×** |

Con validación cruzada agrupada por la fuente de cada texto (**«Keep groups together:»**):

| Modelo | Exactitud balanceada | Tiempo |
|---|---|---|
| Regresión logística (C = 1) | 0,293 | 0,7 min |
| SVM de Orange (núcleo lineal) | **0,332** | 9,4 min |
| Linear SVM (liblinear) | 0,322 | **0,25 min** |

- **Linear SVM (liblinear) queda a 0,010 de la SVM de Orange y es ~38 veces más rápida.**
- **Dos rasgos de la SVM de Orange la mejoraron:** el escalado por rango y el uno contra uno. Con ellos, la exactitud balanceada subió de 0,287 a 0,322: +0,035, con un IC 95 % de [+0,021, +0,049] por bootstrap de grupos.
- **Está medido con un solo conjunto de datos de texto.**

## Cómo funciona
- **`motor.py` (sin Qt):**
  - Un solo grupo de procesos (loky) para toda la sesión de Orange, con 1 hilo de BLAS por proceso.
  - Un candado deja correr un trabajo paralelo a la vez. Si dos widgets arrancan juntos, el segundo espera, y se puede cancelar mientras espera.
  - Cancelar mata los procesos, así que no queda nada corriendo.
  - La tabla se guarda una vez en disco y cada proceso la abre mapeada, en lugar de recibir una copia en cada tarea.
- **`evaluacion.py` (sin Qt):** reproduce el código de Orange 3.40 y solo cambia el bucle en serie por el motor. Abarca:
  - Validation y TestOnTestData;
  - `_search` de Parameter Fitter;
  - `permutation` de Permutation Plot.
- **Cada widget hereda del original y cambia un solo punto:**
  - **Test and Score:** el método que manda a correr la evaluación (`__submit`);
  - **Parameter Fitter y Permutation Plot:** la tarea que arrancan (`start`);
  - **Random Forest:** `create_learner`.
- **Un Corpus va a los procesos como una `Table` común,** con el mismo dominio y las mismas matrices.
  - Orange transforma los datos columna por columna (`SklImpute`, el `Scale` de la SVM). Con un Corpus de Orange3-Text, arma un Corpus nuevo por columna: con un Bag of Words de 130.000 columnas, eso es casi todo el tiempo.
  - Los modelos de Orange no usan nada propio del Corpus, así que los resultados no cambian. Las salidas siguen saliendo del Corpus original.
  - No se convierte si Test and Score tiene preprocesador o si algún modelo no es de Orange.
- **Dentro de un proceso, un modelo con `n_jobs` propio usa 1,** por ejemplo el Random Forest paralelo: los núcleos ya se reparten afuera.

## Riesgos y límites
- **Probado con Orange 3.40.**
  - El add-on se apoya en partes internas de Orange: `Validation`, `_mp_worker`, la cadena de funciones que arma Test and Score y los nombres de las funciones `run`.
  - Con otra versión de Orange, los widgets paralelos corren el camino original en serie y lo avisan (`evaluacion.VERSIONES_PROBADAS`). En Test and Score, «Keep groups together» no está disponible en ese caso, y un segundo aviso dice que la validación cruzada es la común. Linear SVM no depende de partes internas de Orange.
  - Las versiones nuevas del add-on van a seguir a las de Orange. Si una versión de Orange quita algo que el add-on importa, los widgets podrían no cargar hasta que salga la actualización.
- **Heredar de widgets:** Orange 3.40 avisa que «será desactivado en el futuro».
  - El aviso se silencia solo para las 4 clases paralelas. Linear SVM hereda de `OWBaseLearner`, que está abierta a la herencia.
  - Si una versión futura lo prohíbe, los widgets se reescriben por composición.
- **Permutation Plot:** los barajes son los del original (semilla 0, encadenados).
  - El resultado es idéntico si el modelo no usa el generador global de numpy entre un baraje y otro, como la regresión logística o un Random Forest con semilla fija.
  - Con un modelo que sí lo usa, el original tampoco es reproducible, y la versión paralela fija una semilla por permutación.
- **Probabilidades de la SVM de Orange:** el `SVC(probability=True)` que arma Orange no fija `random_state`.
  - Dos corridas del mismo Test and Score original dan las mismas clases predichas y probabilidades distintas (hasta 0,07 en una prueba con texto).
  - Por eso, en la SVM de Orange la identidad con el original vale para las clases, no para las probabilidades.
- **Preprocesador en validación cruzada:** cada proceso lo aplica a su conjunto de entrenamiento, como el original lo aplica a cada uno. Un preprocesador que use el generador global sin semilla puede dar otro resultado, como ya pasa entre dos corridas del original.

## Pruebas
```bash
pip install -e ".[test]"
python -m pytest tests/
```
Son 50 pruebas:
- **identidad con el original** en todas las formas de muestreo, con preprocesador, con un modelo que falla, con un `Corpus` de Orange3-Text, en Parameter Fitter y en Permutation Plot;
- **Corpus como Table:**
  - los procesos reciben una `Table` y los resultados son idénticos;
  - las salidas siguen siendo el Corpus;
  - no se convierte con preprocesador ni con un modelo ajeno;
- **Linear SVM:**
  - las probabilidades suman 1 y su clase más probable es la predicha;
  - es determinista;
  - funciona en regresión y con una clase de un solo ejemplo;
  - un Bag of Words no genera variables por columna;
  - el escalado es igual al de Orange;
- **motor:** orden, errores, cancelación sin procesos huérfanos y dos trabajos a la vez;
- **widgets:** con el marco de pruebas de Orange, incluidas la validación cruzada agrupada y el paso a serie con un Orange no probado.

## Licencia
GPL-3.0 o posterior; ver [LICENSE](https://github.com/Krl05oP11/orange3-paralelo/blob/main/LICENSE). Copyright © 2026 Carlos Ponce Schaller.

Contiene código adaptado de Orange 3.40, © Bioinformatics Laboratory, Facultad de Computación e Informática, Universidad de Liubliana, bajo GPL-3.0 o posterior. Se modificó en octubre de 2026 para repartir el trabajo entre procesos:
- `evaluacion.py` toma partes de `Orange/evaluation/testing.py`, `owparameterfitter.py` y `owpermutationplot.py`;
- los íconos de los widgets son los de Orange.

Desarrollado por **Schaller & Ponce**.
