"""Identidad: la evaluación repartida da lo mismo que la de Orange 3.40, en todas las formas de muestreo."""
import numpy as np
import pytest

from Orange.classification import LogisticRegressionLearner
from Orange.data import Table
from Orange.evaluation import (CrossValidation, CrossValidationFeature, LeaveOneOut, ShuffleSplit, TestOnTestData,
                               TestOnTrainingData)
from Orange.modelling import RandomForestLearner
from Orange.preprocess import Normalize
from Orange.regression import LinearRegressionLearner
from Orange.widgets.evaluate import owparameterfitter, owpermutationplot

from orangecontrib.paralelo import evaluacion, motor

N = 2


class FallaLearner(LogisticRegressionLearner):
    """Un modelo que falla: el original guarda la excepción en results.failed."""
    name = "falla"

    def fit(self, X, Y, W=None):
        raise ValueError("falla a propósito")


def iguales(a, b):
    assert np.array_equal(a.row_indices, b.row_indices) and np.array_equal(a.actual, b.actual, equal_nan=True)
    assert [type(f) for f in a.failed] == [type(f) for f in b.failed]
    ok = [not f for f in a.failed]  # la fila de un modelo que falló queda sin inicializar en los dos
    assert np.array_equal(a.predicted[ok], b.predicted[ok], equal_nan=True)
    if a.probabilities is not None:
        assert np.allclose(a.probabilities[ok], b.probabilities[ok], atol=1e-12, equal_nan=True)
    assert list(a.folds) == list(b.folds)
    assert (a.models is None) == (b.models is None)
    if a.models is not None:
        assert a.models.shape == b.models.shape
        assert [type(m) for m in a.models.ravel()] == [type(m) for m in b.models.ravel()]
    assert (a.data is None) == (b.data is None)
    assert all(x is y for x, y in zip(a.learners, b.learners))  # los mismos objetos: Test and Score lo verifica


def learners_clas():
    return [LogisticRegressionLearner(), RandomForestLearner(n_estimators=15, random_state=0), FallaLearner()]


@pytest.mark.parametrize("sampler", [
    CrossValidation(k=5, random_state=42), CrossValidation(k=5, random_state=42, stratified=False),
    ShuffleSplit(n_resamples=4, train_size=0.7, test_size=None, stratified=True, random_state=42),
    CrossValidation(k=3, random_state=42, store_data=True, store_models=True),
], ids=["cv", "cv_sin_estratos", "shuffle", "cv_guarda"])
def test_validacion_clasificacion(sampler):
    data = Table("heart_disease")
    learners = learners_clas()
    a, b = sampler(data, learners), evaluacion.validacion(sampler, data, learners, n_procesos=N)
    iguales(a, b)
    assert isinstance(b.failed[2], ValueError) and not any(b.failed[:2])  # el modelo que falla, como en el original


def test_validacion_con_preprocesador_y_progreso():
    data, sampler, learners = Table("heart_disease"), CrossValidation(k=4, random_state=42), learners_clas()[:2]
    visto = []
    a = sampler(data, learners, Normalize())
    b = evaluacion.validacion(sampler, data, learners, Normalize(), callback=visto.append, n_procesos=N)
    iguales(a, b)
    assert visto[-1] == 1 and all(0 <= v <= 1 for v in visto)


def test_leave_one_out_y_por_variable():
    iris, learners = Table("iris")[::3], learners_clas()[:1]
    iguales(LeaveOneOut()(iris, learners), evaluacion.validacion(LeaveOneOut(), iris, learners, n_procesos=N))
    titanic, learners = Table("titanic"), learners_clas()
    sampler = CrossValidationFeature(feature=titanic.domain["status"])
    iguales(sampler(titanic, learners), evaluacion.validacion(sampler, titanic, learners, n_procesos=N))


def test_validacion_regresion():
    data = Table("housing")
    learners = [LinearRegressionLearner(), RandomForestLearner(n_estimators=15, random_state=0)]
    sampler = CrossValidation(k=5, random_state=42)
    iguales(sampler(data, learners), evaluacion.validacion(sampler, data, learners, n_procesos=N))


def test_sin_suprimir_excepciones():
    data, sampler = Table("heart_disease"), CrossValidation(k=3, random_state=42)
    with pytest.raises(ValueError, match="a propósito"):
        evaluacion.validacion(sampler, data, [FallaLearner()], suppresses_exceptions=False, n_procesos=N)


def test_prueba_separada_y_sobre_entrenamiento():
    data = Table("heart_disease")
    train, test = data[::2], data[1::2]
    learners = learners_clas()
    sampler = TestOnTestData(store_data=True, store_models=True)
    iguales(sampler(train, test, learners, Normalize()),
            evaluacion.prueba_separada(sampler, train, test, learners, Normalize(), n_procesos=N))
    sampler = TestOnTrainingData(store_models=True)
    iguales(sampler(data, learners), evaluacion.prueba_separada(sampler, data, data, learners, n_procesos=N))


def test_busqueda_igual_a_parameter_fitter():
    data = Table("heart_disease")
    learner = RandomForestLearner(random_state=0)
    props = learner.fitted_parameters[0]
    init = learner.get_params(data)
    steps = [3, 6, 9]
    assert owparameterfitter._search(data, learner, props, init, steps) == \
        evaluacion.busqueda(data, learner, props, init, steps, n_procesos=N)


def test_permutaciones_igual_a_permutation_plot():
    data = Table("iris")
    learner = LogisticRegressionLearner()
    a = owpermutationplot.permutation(data, learner, 6)
    b = evaluacion.permutaciones(data, learner, 6, n_procesos=N)
    assert np.allclose(a[0], b[0]) and a[-1] == b[-1]
    for x, y in zip(a[1:-1], b[1:-1]):
        assert np.allclose(x, y, atol=1e-12)


def test_n_jobs_del_modelo_queda_en_1_adentro():
    learner = RandomForestLearner(n_estimators=5, n_jobs=4)
    copia = evaluacion._un_hilo(learner)
    assert copia.kwargs["n_jobs"] == 1 and copia.get_learner("classification").params["n_jobs"] == 1
    assert learner.kwargs["n_jobs"] == 4  # el original no se toca


def corpus_bow():
    """Bag of Words de book-excerpts (140 textos) con 300 de sus 10.865 palabras y una variable «grupo» de 4 valores:
    sigue siendo un Corpus. Con todas las palabras, la referencia en serie tarda ~35 s."""
    text = pytest.importorskip("orangecontrib.text")
    from orangecontrib.text.vectorization import BowVectorizer
    from Orange.data import DiscreteVariable, Domain
    bow = BowVectorizer().transform(text.Corpus.from_file("book-excerpts"))
    bow = bow.transform(Domain(bow.domain.attributes[:300], bow.domain.class_var, bow.domain.metas))
    bow = bow.add_column(DiscreteVariable("grupo", values=("a", "b", "c", "d")), np.arange(len(bow)) % 4,
                         to_metas=True)
    assert evaluacion._es_corpus(bow) and not evaluacion._es_corpus(Table("iris"))
    return bow


@pytest.fixture
def compartidos(monkeypatch):
    """Lo que se manda a los procesos (un Corpus tiene que llegar como una Table común)."""
    vistos, original = [], motor.compartido

    def espia(obj):
        vistos.extend(obj if isinstance(obj, tuple) else [obj])
        return original(obj)

    monkeypatch.setattr(motor, "compartido", espia)
    return vistos


def test_corpus_de_orange_text():
    """Un Corpus (subclase de Table, con estado propio) sobrevive a guardarlo, mapearlo y cortarlo en los procesos."""
    from Orange.classification import NaiveBayesLearner
    bow, learners = corpus_bow(), [NaiveBayesLearner()]
    for sampler in (CrossValidation(k=5, random_state=42, store_data=True),
                    evaluacion.CVAgrupada(k=5, grupo=bow.domain.metas[0], random_state=42)):
        iguales(sampler(bow, learners), evaluacion.validacion(sampler, bow, learners, n_procesos=N))


@pytest.mark.parametrize("sampler", [
    CrossValidation(k=5, random_state=42, store_data=True, store_models=True),
    CrossValidation(k=5, random_state=42, stratified=False),
    ShuffleSplit(n_resamples=3, train_size=0.7, test_size=None, stratified=True, random_state=42),
    LeaveOneOut(), "por_variable",
], ids=["cv_guarda", "cv_sin_estratos", "shuffle", "loo", "por_variable"])
def test_p6_corpus_como_tabla_en_validacion(sampler, compartidos):
    bow = corpus_bow()
    if sampler == "por_variable":
        sampler = CrossValidationFeature(feature=bow.domain["grupo"])
    data = bow[::4] if isinstance(sampler, LeaveOneOut) else bow
    learners = [LogisticRegressionLearner(), RandomForestLearner(n_estimators=15, random_state=0)]
    a, b = sampler(data, learners), evaluacion.validacion(sampler, data, learners, n_procesos=N)
    iguales(a, b)
    assert [type(x) for x in compartidos] == [Table]  # a los procesos va una Table común…
    if sampler.store_data:
        assert b.data is data  # …y las salidas siguen saliendo del Corpus
    if sampler.store_models:
        assert np.array_equal(b.models[0][0](data[:5]), a.models[0][0](data[:5]))


def test_p6_corpus_como_tabla_en_prueba_separada(compartidos):
    bow = corpus_bow()
    learners = [LogisticRegressionLearner(), RandomForestLearner(n_estimators=15, random_state=0)]
    sampler = TestOnTestData(store_data=True)
    b = evaluacion.prueba_separada(sampler, bow[::2], bow[1::2], learners, n_procesos=N)
    iguales(sampler(bow[::2], bow[1::2], learners), b)
    assert [type(x) for x in compartidos] == [Table, Table] and evaluacion._es_corpus(b.data)


def test_p6_sin_conversion_con_preprocesador_o_modelo_ajeno(compartidos):
    """Un preprocesador de Test and Score o un modelo que no es de Orange podrían depender del Corpus: se respeta."""
    from Orange.preprocess import Continuize
    bow, sampler = corpus_bow(), CrossValidation(k=3, random_state=42)
    learners = [LogisticRegressionLearner()]
    iguales(sampler(bow, learners, Continuize()),
            evaluacion.validacion(sampler, bow, learners, Continuize(), n_procesos=N))
    ajeno = [FallaLearner()]
    iguales(sampler(bow, ajeno), evaluacion.validacion(sampler, bow, ajeno, n_procesos=N))
    assert len(compartidos) == 2 and all(evaluacion._es_corpus(x) for x in compartidos)


def test_p6_corpus_en_parameter_fitter_y_permutation_plot(compartidos):
    bow = corpus_bow()
    learner = RandomForestLearner(random_state=0)
    props, init = learner.fitted_parameters[0], learner.get_params(bow)
    assert owparameterfitter._search(bow, learner, props, init, [3, 6]) == \
        evaluacion.busqueda(bow, learner, props, init, [3, 6], n_procesos=N)
    a = owpermutationplot.permutation(bow, LogisticRegressionLearner(), 4)
    b = evaluacion.permutaciones(bow, LogisticRegressionLearner(), 4, n_procesos=N)
    for x, y in zip(a[:-1], b[:-1]):
        assert np.allclose(x, y, atol=1e-12)
    assert [type(x) for x in compartidos] == [Table, Table]
