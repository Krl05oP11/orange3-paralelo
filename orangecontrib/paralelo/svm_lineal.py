"""SVM lineal con liblinear (LinearSVC / LinearSVR) para Orange: rápido con texto, determinista y con probabilidades
calibradas que coinciden con la clase predicha. Este módulo no importa Qt.

Frente al SVM de Orange (libsvm, `SVMLearner`):
- liblinear resuelve el caso lineal directamente: con un Bag of Words de 130.000 columnas, ~1 s por pliegue;
- la clase predicha es el argmax de las probabilidades (en el SVC de Orange, la votación uno contra uno puede elegir
  otra clase) y dos ajustes dan lo mismo (el SVC de Orange calibra con un generador sin semilla);
- con datos dispersos no imputa y escala adentro del modelo: divide cada columna por su rango (máximo − mínimo, sin
  centrar), igual que el Scale de AdaptiveNormalize, pero sin crear una variable por columna, que es lo que vuelve lento
  a Orange con un Corpus;
- multiclase uno contra uno, como libsvm.
Con un corpus de texto de 18 clases (2.699 textos × 130.047 columnas de Bag of Words, CV agrupada), el escalado y el uno
contra uno suben la BA de 0,287 a 0,322 (la SVM de Orange: 0,332 en 9,4 min; esta, 0,25 min).
"""
import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, LinearSVR

from Orange.classification import SklLearner as SklClasificacion
from Orange.modelling import SklFitter
from Orange.base import SklLearner
from Orange.preprocess import AdaptiveNormalize, SklImpute
from Orange.preprocess.preprocess import Preprocess
from Orange.regression import SklLearner as SklRegresion

N_CALIBRACION = 5  # pliegues de la calibración (sigmoide)


class ImputarSiFalta(Preprocess):
    """SklImpute, salvo con datos dispersos sin faltantes (un Bag of Words), donde crearía una variable por columna."""

    def __call__(self, data):
        if sp.issparse(data.X) and not np.isnan(data.X.data).any():
            return data
        return SklImpute()(data)


class NormalizarSiDenso(Preprocess):
    """Con datos densos, lo de las SVM de Orange (AdaptiveNormalize); con dispersos, nada: el mismo escalado por rango lo
    hace SVCLinealCalibrado adentro (factores_rango), sin el Scale por columna que es lo lento con un Corpus."""

    def __call__(self, data):
        return data if sp.issparse(data.X) else AdaptiveNormalize()(data)


# los de Orange (HasClass, Continuize, RemoveNaNColumns, SklImpute) con SklImpute cambiado, y la normalización
preprocesadores = [ImputarSiFalta() if isinstance(p, SklImpute) else p for p in SklLearner.preprocessors] + \
                  [NormalizarSiDenso()]


def factores_rango(X):
    """1 / (máximo − mínimo) de cada columna de una matriz dispersa, con los ceros implícitos; 1 si el rango es < 1e-15.
    Es el Scale(NoCentering, Span) que AdaptiveNormalize aplica a los datos dispersos en las SVM de Orange."""
    s = X.max(axis=0).toarray().ravel() - X.min(axis=0).toarray().ravel()
    s[s < 1e-15] = 1
    return 1 / s


def _limpio(X):
    """Un NaN que llegue al predecir en datos dispersos cuenta como 0 (en los densos ya se imputó)."""
    if sp.issparse(X) and np.isnan(X.data).any():
        X = X.copy()
        X.data = np.nan_to_num(X.data)
    return X


class SVCLinealCalibrado(ClassifierMixin, BaseEstimator):
    """LinearSVC uno contra uno, con los datos dispersos escalados por rango y los puntajes calibrados por una sigmoide
    (validación interna de 5 pliegues estratificados, sin barajar). Si alguna clase tiene menos de 2 ejemplos no se puede calibrar: las probabilidades son el softmax de los
    puntajes."""

    def __init__(self, C=1.0, class_weight=None, max_iter=5000, tol=1e-4):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    def _escalado(self, X):
        X = _limpio(X)
        return sp.csr_matrix(X.multiply(self.factores_)) if self.factores_ is not None else X

    def fit(self, X, y, sample_weight=None):
        self.factores_ = factores_rango(_limpio(X)) if sp.issparse(X) else None  # densos: ya normalizados afuera
        X = self._escalado(X)
        self.classes_, cuentas = np.unique(y, return_counts=True)
        svc = OneVsOneClassifier(LinearSVC(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter,
                                           tol=self.tol, random_state=0))
        k = min(N_CALIBRACION, cuentas.min())
        self.calibrado_ = k >= 2
        if self.calibrado_:
            svc = CalibratedClassifierCV(svc, method="sigmoid", cv=StratifiedKFold(k), ensemble=False)
        # OneVsOneClassifier no acepta pesos: con calibración se usan solo para calibrar; sin ella, se ignoran
        pesos = {"sample_weight": sample_weight} if self.calibrado_ and sample_weight is not None else {}
        self.modelo_ = svc.fit(X, y, **pesos)
        return self

    def predict_proba(self, X):
        X = self._escalado(X)
        if self.calibrado_:
            return self.modelo_.predict_proba(X)
        puntajes = self.modelo_.decision_function(X)
        if puntajes.ndim == 1:  # dos clases: un solo puntaje, a favor de la segunda
            puntajes = np.column_stack([-puntajes, puntajes])
        return softmax(puntajes, axis=1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SVCLinealLearner(SklClasificacion):
    __wraps__ = SVCLinealCalibrado
    name = "Linear SVM"
    preprocessors = preprocesadores

    def __init__(self, C=1.0, class_weight=None, max_iter=5000, tol=1e-4, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SVRLinealLearner(SklRegresion):
    __wraps__ = LinearSVR
    name = "Linear SVM"
    preprocessors = preprocesadores

    def __init__(self, C=1.0, epsilon=0.0, max_iter=5000, tol=1e-4, random_state=0, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SVMLinealLearner(SklFitter):
    """Clasificación (SVCLinealCalibrado) o regresión (LinearSVR) según la variable de clase. El Fitter le pasa a cada
    uno solo los parámetros que acepta (class_weight, solo a la clasificación)."""
    __fits__ = {"classification": SVCLinealLearner, "regression": SVRLinealLearner}
    name = "Linear SVM"
