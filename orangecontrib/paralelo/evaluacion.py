"""La evaluación de Orange 3.40 repartida con el motor, con los mismos resultados que el original.

- `validacion`: Validation.__call__ (CrossValidation, CrossValidationFeature, LeaveOneOut, ShuffleSplit), una tarea por
  par (pliegue × modelo).
- `prueba_separada`: TestOnTestData.__call__ (y TestOnTrainingData), una tarea por modelo.
- `busqueda`: _search de Parameter Fitter, una tarea por punto de la grilla.
- `permutaciones`: permutation de Permutation Plot, una tarea por permutación (los barajes se generan antes, en orden).
- `evaluar_muestreo`: lo que corre Test and Score; decide serie o paralelo y llama a `validacion` o `prueba_separada`.
- `CVAgrupada`: validación cruzada que no separa los grupos de una variable (p. ej. los textos de un mismo prompt).

Las funciones públicas corren en el proceso de Orange y asumen que ya se decidió repartir (`motor.conviene_paralelo`);
las que empiezan con _ corren en los procesos de trabajo (salvo `_para_procesos`). Este módulo no importa Qt.

Contiene código de Orange 3.40 (Orange/evaluation/testing.py, Orange/widgets/evaluate/owparameterfitter.py y
owpermutationplot.py), © Bioinformatics Laboratory, FRI UL, GPL-3.0-or-later, modificado en octubre de 2026 para
repartir el trabajo entre procesos.
"""
import copy

import numpy as np
from scipy.stats import linregress, spearmanr

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from Orange.evaluation import (CrossValidation, CrossValidationFeature, LeaveOneOut, ShuffleSplit, TestOnTestData,
                               TestOnTrainingData)
from Orange.evaluation.scoring import AUC, R2
from Orange.data import Table
from Orange.evaluation.testing import Results, _identity, _mp_worker
from Orange.modelling import Fitter
from Orange.version import short_version

from . import motor

N_FOLD = 7  # el de owparameterfitter y owpermutationplot
# Versiones de Orange con las que se probó la identidad con el original (este módulo copia partes internas). Con otra,
# los widgets corren el original en serie y avisan.
VERSIONES_PROBADAS = ("3.40",)
ORANGE_PROBADO = ".".join(short_version.split(".")[:2]) in VERSIONES_PROBADAS


def _es_corpus(data):
    """¿Es un Corpus de Orange3-Text? Sin importar orangecontrib.text, que es opcional."""
    return any(c.__name__ == "Corpus" and c.__module__.startswith("orangecontrib.text") for c in type(data).__mro__)


def _de_orange(learner):
    return type(learner).__module__.startswith(("Orange.", "orangecontrib.paralelo."))


def _para_procesos(data, learners, preprocessor=None):
    """Los datos que se mandan a los procesos: un Corpus va como una Table común con el mismo dominio y las mismas
    matrices.

    Orange transforma los datos columna por columna (SklImpute, el Scale de la SVM) y, con un Corpus, arma un Corpus
    nuevo por columna: con un Bag of Words de 130.000 columnas, un pliegue de SVM pasa de ~5 min a ~2 h. Los modelos de
    Orange no usan nada propio del Corpus, así que los resultados no cambian. No se convierte si hay preprocesador o
    algún modelo que no es de Orange: podrían depender del Corpus.
    """
    if not _es_corpus(data) or preprocessor is not None or not all(map(_de_orange, learners)):
        return data
    return Table.from_numpy(data.domain, data.X, data.Y, data.metas, data.W, data.attributes, data.ids)


def _un_hilo(learner):
    """Copia del modelo con n_jobs = 1 (p. ej. un Random Forest paralelo): los núcleos ya se reparten entre procesos."""
    if isinstance(learner, Fitter):
        if learner.kwargs.get("n_jobs", 1) != 1:
            learner = copy.copy(learner)
            learner.kwargs = {**learner.kwargs, "n_jobs": 1}
            learner._Fitter__learners = dict.fromkeys(learner._Fitter__learners)  # se rehacen con los kwargs nuevos
    elif isinstance(getattr(learner, "params", None), dict) and learner.params.get("n_jobs", 1) != 1:
        learner = copy.copy(learner)
        learner.params = {**learner.params, "n_jobs": 1}
    return learner


def _resultados(sampler, domain, learners, n_folds, partes, **campos):
    """Results como al final de Validation.__call__ / TestOnTestData.__call__."""
    results = Results(domain=domain, learners=learners, score_by_folds=sampler.score_by_folds,
                      train_time=np.zeros((len(learners),)), test_time=np.zeros((len(learners),)), **campos)
    if sampler.store_models:
        results.models = np.tile(None, (n_folds, len(learners)))
    sampler._collect_part_results(results, partes)
    return results


def _parte(ruta, fold_i, train_i, test_i, learner_i, learner, preprocessor, store_models, suppresses_exceptions):
    data = motor.cargar(ruta)
    return _mp_worker(fold_i, preprocessor(data[train_i]), data[test_i], learner_i, _un_hilo(learner), store_models,
                      suppresses_exceptions)


def validacion(sampler, data, learners, preprocessor=None, *, callback=None, suppresses_exceptions=True,
               n_procesos=motor.N_DEFECTO, cancelado=None):
    """sampler(data, learners, preprocessor, callback=..., suppresses_exceptions=...) con el bucle repartido.

    El preprocesador se aplica en cada proceso a su conjunto de entrenamiento, como el original lo aplica a cada uno.
    """
    compartir = _para_procesos(data, learners, preprocessor)
    preprocessor = preprocessor or _identity
    callback = callback or _identity
    indices = sampler.get_indices(data)
    folds, row_indices, actual = sampler.prepare_arrays(data, indices)
    with motor.compartido(compartir) as ruta:
        tareas = [(ruta, fold_i, train_i, test_i, learner_i, learner, preprocessor, sampler.store_models,
                   suppresses_exceptions)
                  for fold_i, (train_i, test_i) in enumerate(indices) for learner_i, learner in enumerate(learners)]
        partes = motor.ejecutar(_parte, tareas, n_procesos, lambda hechas, total: callback(0.99 * hechas / total),
                                cancelado)
    callback(1)
    return _resultados(sampler, data.domain, learners, len(indices), partes, data=data if sampler.store_data else None,
                       nrows=len(row_indices), row_indices=row_indices, folds=folds, actual=actual)


def _parte_separada(ruta, learner_i, learner, store_models, suppresses_exceptions):
    train, test = motor.cargar(ruta)
    return _mp_worker(0, train, test, learner_i, _un_hilo(learner), store_models, suppresses_exceptions)


def prueba_separada(sampler, data, test_data, learners, preprocessor=None, *, callback=None,
                    suppresses_exceptions=True, n_procesos=motor.N_DEFECTO, cancelado=None):
    """TestOnTestData.__call__ (también para TestOnTrainingData, con test_data = data) con un proceso por modelo."""
    callback = callback or _identity
    train = (preprocessor or _identity)(data)  # una vez, como el original
    compartir = (_para_procesos(train, learners, preprocessor), _para_procesos(test_data, learners, preprocessor))
    with motor.compartido(compartir) as ruta:
        tareas = [(ruta, learner_i, learner, sampler.store_models, suppresses_exceptions)
                  for learner_i, learner in enumerate(learners)]
        partes = motor.ejecutar(_parte_separada, tareas, n_procesos, lambda hechas, total: callback(hechas / total),
                                cancelado)
    callback(1)
    return _resultados(sampler, test_data.domain, learners, 1, partes, data=test_data if sampler.store_data else None,
                       nrows=len(test_data), row_indices=np.arange(len(test_data)), folds=(Ellipsis,),
                       actual=test_data.Y)


class CVAgrupada(CrossValidation):
    """k pliegues sin separar los grupos de la variable `grupo`: StratifiedGroupKFold (estratificado por la clase,
    barajado con random_state) o GroupKFold. Los valores faltantes forman un grupo."""

    def __init__(self, k=10, grupo=None, stratified=True, random_state=0, store_data=False, store_models=False):
        super().__init__(k=k, stratified=stratified, random_state=random_state, store_data=store_data,
                         store_models=store_models)
        self.grupo = grupo

    def get_indices(self, data):
        grupos = data.get_column(self.grupo)
        if self.stratified and data.domain.has_discrete_class:
            splitter = StratifiedGroupKFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        else:
            splitter = GroupKFold(n_splits=self.k)
        return list(splitter.split(np.zeros((len(data), 1)), data.Y, groups=grupos))


def n_pliegues(sampler, data):
    """Cuántos conjuntos de prueba arma el muestreo (sin armarlos)."""
    if isinstance(sampler, TestOnTestData):
        return 1
    if isinstance(sampler, CrossValidationFeature):
        return len(sampler.feature.values)
    if isinstance(sampler, ShuffleSplit):
        return sampler.n_resamples
    if isinstance(sampler, LeaveOneOut):
        return len(data)
    return sampler.k


def evaluar_muestreo(sampler, args, n_procesos, callback=None):
    """sampler(*args, callback=callback), como lo arma Test and Score, repartido si conviene.

    args es (data, learners, preprocessor), o (data, test_data, learners, preprocessor) con TestOnTestData. Cancelar:
    el callback de Test and Score lanza UserInterrupt cuando se pide; acá también se lo consulta mientras se espera.
    """
    callback = callback or _identity
    if isinstance(sampler, TestOnTrainingData):
        (data, learners, preprocessor), test_data = args, args[0]
    elif isinstance(sampler, TestOnTestData):
        data, test_data, learners, preprocessor = args
    else:
        (data, learners, preprocessor), test_data = args, None
    n_tareas = n_pliegues(sampler, data) * len(learners)
    if not motor.conviene_paralelo(n_procesos, n_tareas, motor.celdas(data) * n_tareas):
        return sampler(*args, callback=callback)
    ultimo = [0.0]

    def avance(p):
        ultimo[0] = p
        callback(p)

    def cancelado():
        callback(ultimo[0])  # lanza si se pidió cancelar
        return False

    if test_data is None:
        return validacion(sampler, data, learners, preprocessor, callback=avance, n_procesos=n_procesos,
                          cancelado=cancelado)
    return prueba_separada(sampler, data, test_data, learners, preprocessor, callback=avance, n_procesos=n_procesos,
                           cancelado=cancelado)


def _validar(data, learner):
    """_validate de owparameterfitter y owpermutationplot (Orange 3.40), sin callback."""
    scorer = AUC if data.domain.has_discrete_class else R2
    res = TestOnTrainingData()(data, [learner], suppresses_exceptions=False)
    res_cv = CrossValidation(k=N_FOLD)(data, [learner], suppresses_exceptions=False)
    return scorer(res)[0], scorer(res_cv)[0]


def _punto(ruta, tipo, params):
    return _validar(motor.cargar(ruta), _un_hilo(tipo(**params)))


def busqueda(data, learner, fitted_parameter_props, initial_parameters, steps, progreso=None, *,
             n_procesos=motor.N_DEFECTO, cancelado=None):
    """_search de owparameterfitter: (puntajes, nombre del puntaje, etiqueta del parámetro), un proceso por punto."""
    name = fitted_parameter_props.name
    with motor.compartido(_para_procesos(data, [learner])) as ruta:
        tareas = [(ruta, type(learner), {**initial_parameters, name: value}) for value in steps]
        puntajes = motor.ejecutar(_punto, tareas, n_procesos, progreso, cancelado)
    scorer = AUC if data.domain.has_discrete_class else R2
    return list(zip(steps, puntajes)), scorer.name, fitted_parameter_props.label


def _permutacion(ruta, learner, y, semilla):
    data = motor.cargar(ruta)
    if y is not None:
        data = data.copy()
        with data.unlocked(data.Y):
            data.Y[:] = y
    np.random.seed(semilla)  # reproducible aunque el modelo use el generador global (ver README)
    return _validar(data, _un_hilo(learner))


def permutaciones(data, learner, n_perm=100, progreso=None, *, n_procesos=motor.N_DEFECTO, cancelado=None):
    """permutation de owpermutationplot. Los barajes son los del original (semilla 0, encadenados) si el modelo no usa
    el generador global de numpy entre uno y otro."""
    np.random.seed(0)
    data_perm = data.copy()
    ys = []
    for _ in range(n_perm):
        np.random.shuffle(data_perm.Y)
        ys.append(np.array(data_perm.Y))
    with motor.compartido(_para_procesos(data, [learner])) as ruta:
        tareas = [(ruta, learner, None, 0)] + [(ruta, learner, y, i + 1) for i, y in enumerate(ys)]
        puntajes = motor.ejecutar(_permutacion, tareas, n_procesos, progreso, cancelado)
    scores_tr, scores_cv = [p[0] for p in puntajes], [p[1] for p in puntajes]
    correlations = np.abs([100.0] + [spearmanr(data.Y, y)[0] * 100 for y in ys])
    res_tr = linregress([correlations[0], np.mean(correlations[1:])], [scores_tr[0], np.mean(scores_tr[1:])])
    res_cv = linregress([correlations[0], np.mean(correlations[1:])], [scores_cv[0], np.mean(scores_cv[1:])])
    scorer = AUC if data.domain.has_discrete_class else R2
    return (correlations, scores_tr, res_tr.intercept, res_tr.slope,
            scores_cv, res_cv.intercept, res_cv.slope, scorer.name)
