"""SVM lineal (liblinear): probabilidades coherentes con la clase predicha, determinista, y sin variables por columna con
un Bag of Words."""
import numpy as np
import pytest
import scipy.sparse as sp

from Orange.data import Domain, Table
from Orange.evaluation import CrossValidation
from Orange.preprocess import Scale
from sklearn.multiclass import OneVsOneClassifier

from orangecontrib.paralelo.svm_lineal import SVCLinealCalibrado, SVMLinealLearner, factores_rango


@pytest.mark.parametrize("nombre", ["iris", "heart_disease"])  # heart_disease tiene faltantes
def test_probabilidades_y_clase_predicha(nombre):
    data = Table(nombre)
    model = SVMLinealLearner()(data)
    clase, probs = model(data, model.ValueProbs)
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.array_equal(clase, probs.argmax(axis=1))
    assert model.skl_model.calibrado_
    res = CrossValidation(k=5, random_state=0)(data, [SVMLinealLearner()])
    assert not res.failed[0] and np.mean(res.predicted[0] == res.actual) > 0.75


def test_determinista():
    data = Table("heart_disease")
    a, b = SVMLinealLearner()(data)(data, 1), SVMLinealLearner()(data)(data, 1)
    assert np.array_equal(a, b)


def test_regresion():
    data = Table("housing")
    learner = SVMLinealLearner(C=0.5, class_weight="balanced")  # class_weight no llega a la regresión
    assert "class_weight" not in learner.get_learner("regression").params
    res = CrossValidation(k=5, random_state=0)(data, [learner])
    assert not res.failed[0] and np.corrcoef(res.predicted[0], res.actual)[0, 1] > 0.7


def test_clase_con_un_ejemplo_usa_softmax():
    iris = Table("iris")
    data = Table.concatenate([iris[:50], iris[50:51], iris[100:]])
    model = SVMLinealLearner()(data)
    assert not model.skl_model.calibrado_
    clase, probs = model(data, model.ValueProbs)
    assert np.allclose(probs.sum(axis=1), 1) and np.array_equal(clase, probs.argmax(axis=1))


def test_bag_of_words_sin_variables_por_columna():
    """Con un Corpus disperso sin faltantes no se imputa ni se escala afuera: el dominio del modelo usa las columnas tal
    cual (lo que en el SVM de Orange crea una variable por columna, y con un Corpus lo vuelve lento)."""
    text = pytest.importorskip("orangecontrib.text")
    from orangecontrib.text.vectorization import BowVectorizer
    bow = BowVectorizer().transform(text.Corpus.from_file("book-excerpts"))
    assert sp.issparse(bow.X)
    model = SVMLinealLearner()(bow)
    assert all(a.compute_value is b.compute_value for a, b in zip(model.domain.attributes, bow.domain.attributes))
    clase, probs = model(bow[:10], model.ValueProbs)
    assert np.allclose(probs.sum(axis=1), 1) and np.array_equal(clase, probs.argmax(axis=1))


def test_nan_disperso_al_predecir():
    text = pytest.importorskip("orangecontrib.text")
    from orangecontrib.text.vectorization import BowVectorizer
    bow = BowVectorizer().transform(text.Corpus.from_file("book-excerpts"))
    model = SVMLinealLearner()(bow)
    X = bow.X[:3].tocsr().copy()
    X.data[:5] = np.nan
    assert np.allclose(model.skl_model.predict_proba(X).sum(axis=1), 1)


def test_disperso_escala_como_las_svm_de_orange():
    """Con datos dispersos, el modelo divide cada columna por su rango (lo de AdaptiveNormalize en las SVM de
    Orange), con los rangos del entrenamiento, y es uno contra uno como libsvm."""
    text = pytest.importorskip("orangecontrib.text")
    from orangecontrib.text.vectorization import BowVectorizer
    bow = BowVectorizer().transform(text.Corpus.from_file("book-excerpts"))
    sub = Table.from_numpy(Domain(bow.domain.attributes[:300], bow.domain.class_var), bow.X[:, :300], bow.Y)
    orange = Scale(center=Scale.NoCentering, scale=Scale.Span)(sub).X
    orange = orange.toarray() if sp.issparse(orange) else orange
    assert np.allclose(orange, sub.X.multiply(factores_rango(sub.X)).toarray(), atol=1e-12)
    model = SVMLinealLearner()(bow[:80])
    assert np.array_equal(model.skl_model.factores_, factores_rango(bow[:80].X))
    assert isinstance(model.skl_model.modelo_.estimator, OneVsOneClassifier)


def test_pesos_y_clase_con_un_ejemplo():
    """OneVsOneClassifier no acepta pesos: calibrado los usa para calibrar; sin calibrar (una clase con un ejemplo), se
    ignoran sin error. Se prueba el estimador: en Orange 3.40, AdaptiveNormalize falla con datos densos con pesos (también
    en la SVM de Orange)."""
    iris = Table("iris")
    for X, y in ((iris.X, iris.Y), (np.vstack([iris.X[:51], iris.X[100:]]), np.hstack([iris.Y[:51], iris.Y[100:]]))):
        m = SVCLinealCalibrado().fit(X, y, sample_weight=np.ones(len(y)))
        assert m.calibrado_ == (len(y) == 150) and np.allclose(m.predict_proba(X).sum(axis=1), 1)
