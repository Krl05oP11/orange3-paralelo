"""Los 4 widgets con el marco de pruebas de Orange: mismas salidas que el original y el camino paralelo de verdad usado."""
from contextlib import contextmanager

import numpy as np

from Orange.classification import LogisticRegressionLearner
from Orange.data import Domain, StringVariable, Table
from Orange.modelling import RandomForestLearner
from Orange.widgets.evaluate import owparameterfitter, owpermutationplot
from Orange.widgets.evaluate.owtestandscore import OWTestAndScore
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.paralelo import evaluacion, motor
from orangecontrib.paralelo.widgets.owparameterfitter_paralelo import OWParameterFitterParalelo
from orangecontrib.paralelo.widgets.owpermutationplot_paralelo import OWPermutationPlotParalelo
from orangecontrib.paralelo.widgets.owrandomforest_paralelo import OWRandomForestParalelo
from orangecontrib.paralelo.widgets.owsvmlineal import OWSVMLineal
from orangecontrib.paralelo.widgets.owtestandscore_paralelo import OWTestAndScoreParalelo


@contextmanager
def orange_no_probado():
    """Simula una versión de Orange fuera de evaluacion.VERSIONES_PROBADAS."""
    evaluacion.ORANGE_PROBADO = False
    try:
        yield
    finally:
        evaluacion.ORANGE_PROBADO = True


class ParaleloForzado(WidgetTest):
    """Datos chicos por el camino paralelo (umbral 0), contando las llamadas al motor."""

    def setUp(self):
        self._umbral, motor.UMBRAL_TRABAJO = motor.UMBRAL_TRABAJO, 0
        self._ejecutar, self.llamadas = motor.ejecutar, []

        def espia(*args, **kwargs):
            self.llamadas.append(args[2])  # n_procesos
            return self._ejecutar(*args, **kwargs)

        motor.ejecutar = espia

    def tearDown(self):
        motor.UMBRAL_TRABAJO, motor.ejecutar = self._umbral, self._ejecutar


class TestTestAndScore(ParaleloForzado):
    def resultados(self, cls, data, learners, **attrs):
        # con la versión: sin ella, migrate_settings del original falla y descarta todos los ajustes guardados
        w = self.create_widget(cls, stored_settings={"__version__": cls.settings_version, **attrs})
        self.send_signal(w.Inputs.train_data, data, widget=w)
        for i, learner in enumerate(learners):
            self.send_signal(w.Inputs.learner, learner, i, widget=w)
        return self.get_output(w.Outputs.evaluations_results, widget=w, wait=20000), w

    def test_igual_al_original_en_paralelo(self):
        data, learners = Table("heart_disease"), [LogisticRegressionLearner(), RandomForestLearner(random_state=0)]
        a, _ = self.resultados(OWTestAndScore, data, learners)
        b, w = self.resultados(OWTestAndScoreParalelo, data, learners, n_procesos=2)
        assert self.llamadas and set(self.llamadas) == {2}
        assert np.array_equal(a.predicted, b.predicted) and np.allclose(a.probabilities, b.probabilities, atol=1e-12)
        # la identidad de los modelos la verifica el propio replace_learners del original: si fallara, no habría salida
        assert not w.Warning.sin_paralelo.is_shown()

    def test_un_proceso_es_el_original(self):
        data, learners = Table("heart_disease"), [LogisticRegressionLearner()]
        self.resultados(OWTestAndScoreParalelo, data, learners, n_procesos=1)
        assert not self.llamadas

    def test_cv_agrupada_no_separa_grupos(self):
        heart = Table("heart_disease")
        grupo = StringVariable("grupo")
        dom = Domain(heart.domain.attributes, heart.domain.class_var, heart.domain.metas + (grupo,))
        metas = np.array([[f"g{i // 3}"] for i in range(len(heart))], dtype=object)
        data = Table.from_numpy(dom, heart.X, heart.Y, np.hstack([heart.metas, metas]) if heart.metas.size else metas)
        w = self.create_widget(OWTestAndScoreParalelo, stored_settings={
            "__version__": OWTestAndScoreParalelo.settings_version, "n_procesos": 2, "cv_agrupada": True})
        self.send_signal(w.Inputs.train_data, data, widget=w)
        w.grupo_var = grupo
        w.kfold_changed()
        self.send_signal(w.Inputs.learner, LogisticRegressionLearner(), 0, widget=w)
        res = self.get_output(w.Outputs.evaluations_results, widget=w, wait=20000)
        g = data.get_column(grupo)
        por_pliegue = [set(g[res.row_indices[f]]) for f in res.folds]
        assert len(por_pliegue) == 5 and all(not (a & b) for i, a in enumerate(por_pliegue) for b in por_pliegue[i + 1:])
        assert not w.Warning.sin_grupo.is_shown()

    def test_orange_no_probado_corre_el_original(self):
        data, learners = Table("heart_disease"), [LogisticRegressionLearner()]
        a, _ = self.resultados(OWTestAndScore, data, learners)
        with orange_no_probado():
            b, w = self.resultados(OWTestAndScoreParalelo, data, learners, n_procesos=2)
        assert not self.llamadas and w.Warning.sin_paralelo.is_shown() and not w.Warning.sin_agrupar.is_shown()
        assert np.array_equal(a.predicted, b.predicted) and np.allclose(a.probabilities, b.probabilities, atol=1e-12)

    def test_orange_no_probado_avisa_que_no_agrupa(self):
        with orange_no_probado():
            _, w = self.resultados(OWTestAndScoreParalelo, Table("heart_disease"), [LogisticRegressionLearner()],
                                   n_procesos=2, cv_agrupada=True)
        assert w.Warning.sin_paralelo.is_shown() and w.Warning.sin_agrupar.is_shown()


class TestParameterFitter(ParaleloForzado):
    def test_igual_al_original(self):
        data, learner = Table("heart_disease"), RandomForestLearner(random_state=0)
        w = self.create_widget(OWParameterFitterParalelo, stored_settings={"n_procesos": 2})
        hechos = []
        w.on_done = hechos.append
        self.send_signal(w.Inputs.data, data, widget=w)
        self.send_signal(w.Inputs.learner, learner, widget=w)
        self.wait_until_finished(widget=w, timeout=60000)
        assert self.llamadas == [2] and hechos
        esperado = owparameterfitter._search(data, learner, w.fitted_parameters[w.parameter_index],
                                             w.initial_parameters, w.steps)
        assert hechos[0] == esperado
        assert not w.Warning.sin_paralelo.is_shown()

    def test_orange_no_probado_corre_el_original(self):
        with orange_no_probado():
            w = self.create_widget(OWParameterFitterParalelo, stored_settings={"n_procesos": 2})
            hechos = []
            w.on_done = hechos.append
            self.send_signal(w.Inputs.data, Table("heart_disease"), widget=w)
            self.send_signal(w.Inputs.learner, RandomForestLearner(random_state=0), widget=w)
            self.wait_until_finished(widget=w, timeout=60000)
        assert not self.llamadas and hechos and w.Warning.sin_paralelo.is_shown()


class TestPermutationPlot(ParaleloForzado):
    def test_igual_al_original(self):
        data, learner = Table("iris"), LogisticRegressionLearner()
        w = self.create_widget(OWPermutationPlotParalelo, stored_settings={"n_procesos": 2, "n_permutations": 8})
        hechos = []
        w.on_done = hechos.append
        self.send_signal(w.Inputs.data, data, widget=w)
        self.send_signal(w.Inputs.learner, learner, widget=w)
        self.wait_until_finished(widget=w, timeout=60000)
        assert self.llamadas == [2] and hechos
        esperado = owpermutationplot.permutation(data, learner, 8)
        for x, y in zip(esperado[:-1], hechos[0][:-1]):
            assert np.allclose(x, y, atol=1e-12)
        assert esperado[-1] == hechos[0][-1]
        assert not w.Warning.sin_paralelo.is_shown()

    def test_orange_no_probado_corre_el_original(self):
        with orange_no_probado():
            w = self.create_widget(OWPermutationPlotParalelo, stored_settings={"n_procesos": 2, "n_permutations": 8})
            hechos = []
            w.on_done = hechos.append
            self.send_signal(w.Inputs.data, Table("iris"), widget=w)
            self.send_signal(w.Inputs.learner, LogisticRegressionLearner(), widget=w)
            self.wait_until_finished(widget=w, timeout=60000)
        assert not self.llamadas and hechos and w.Warning.sin_paralelo.is_shown()


class TestRandomForest(WidgetTest):
    def test_n_jobs_en_el_modelo(self):
        w = self.create_widget(OWRandomForestParalelo, stored_settings={"n_procesos": 3})
        learner = self.get_output(w.Outputs.learner, widget=w)
        assert learner.kwargs["n_jobs"] == 3
        assert learner.get_learner("classification").params["n_jobs"] == 3
        assert evaluacion._un_hilo(learner).get_learner("classification").params["n_jobs"] == 1


class TestSVMLineal(ParaleloForzado):
    def test_parametros_en_el_modelo(self):
        w = self.create_widget(OWSVMLineal, stored_settings={"C": 0.5, "balanceadas": True, "max_iter": 300})
        learner = self.get_output(w.Outputs.learner, widget=w)
        params = learner.get_learner("classification").params
        assert (params["C"], params["class_weight"], params["max_iter"]) == (0.5, "balanced", 300)
        assert "class_weight" not in learner.get_learner("regression").params

    def test_en_test_and_score_original_y_paralelo(self):
        data = Table("heart_disease")
        svm = self.create_widget(OWSVMLineal)
        learner = self.get_output(svm.Outputs.learner, widget=svm)
        salidas = []
        for cls, extra in ((OWTestAndScore, {}), (OWTestAndScoreParalelo, {"n_procesos": 2})):
            w = self.create_widget(cls, stored_settings={"__version__": cls.settings_version, **extra})
            self.send_signal(w.Inputs.train_data, data, widget=w)
            self.send_signal(w.Inputs.learner, learner, 0, widget=w)
            salidas.append(self.get_output(w.Outputs.evaluations_results, widget=w, wait=20000))
        a, b = salidas
        assert not a.failed[0] and self.llamadas == [2]
        # determinista: el paralelo da exactamente lo mismo, también en las probabilidades
        assert np.array_equal(a.predicted, b.predicted) and np.allclose(a.probabilities, b.probabilities, atol=1e-12)


def test_cada_modulo_expone_solo_su_widget():
    """El descubrimiento de Orange registra la primera clase de widget de cada módulo: los originales no pueden estar
    en el espacio de nombres (se importa el módulo, no la clase), o se registrarían dos veces."""
    from orangecontrib.paralelo.widgets import (owparameterfitter_paralelo, owpermutationplot_paralelo,
                                                owrandomforest_paralelo, owsvmlineal, owtestandscore_paralelo)
    for m in (owtestandscore_paralelo, owparameterfitter_paralelo, owpermutationplot_paralelo, owrandomforest_paralelo,
              owsvmlineal):
        clases = [v for v in vars(m).values() if hasattr(v, "get_widget_description")]
        assert len(clases) == 1 and clases[0].__module__ == m.__name__, m.__name__
