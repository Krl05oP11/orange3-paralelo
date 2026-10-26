"""Test and Score (paralelo): el widget original, con la evaluación repartida entre procesos.

Hereda de OWTestAndScore y cambia solo el punto donde el original manda a correr la evaluación (__submit): el muestreo
armado por el original se corre con `evaluacion.evaluar_muestreo`. Agrega la validación cruzada agrupada por una
variable (StratifiedGroupKFold), para datos con varias filas por grupo.
"""
from functools import partial

from Orange.data import DiscreteVariable, StringVariable
from Orange.evaluation import CrossValidation
from Orange.evaluation.testing import Validation
from Orange.widgets import gui, settings
from Orange.widgets.evaluate import owtestandscore
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Msg

from orangecontrib.paralelo import evaluacion, motor
from orangecontrib.paralelo.widgets.comun import SIN_PARALELO, caja_procesos


class EstructuraDesconocida(Exception):
    """Orange no probado o la función que arma no es la de 3.40: se corre la original, en serie."""


class OWTestAndScoreParalelo(owtestandscore.OWTestAndScore):
    name = "Test and Score (parallel)"
    description = "Test and Score with the evaluation spread across CPU cores."
    icon = "icons/TestLearners1.svg"
    priority = 10
    keywords = "test and score, cross validation, cv, group, parallel, paralelo"
    replaces = []

    n_procesos = settings.Setting(motor.N_DEFECTO)
    cv_agrupada = settings.Setting(False)
    grupo_var = settings.ContextSetting(None)

    class Warning(owtestandscore.OWTestAndScore.Warning):
        sin_paralelo = Msg(SIN_PARALELO)
        sin_grupo = Msg("No group variable selected: running ordinary cross validation.")
        sin_agrupar = Msg("Groups are not kept together on this Orange version: running ordinary cross validation.")

    def __init__(self):
        super().__init__()
        self.grupo_model = DomainModel(order=DomainModel.MIXED, valid_types=(DiscreteVariable, StringVariable))
        ibox = self.controls.cv_stratified.parentWidget()  # la caja de «Cross validation» del original
        gui.checkBox(ibox, self, "cv_agrupada", "Keep groups together:", callback=self.kfold_changed,
                     tooltip="StratifiedGroupKFold: rows with the same value stay in the same fold.")
        gui.comboBox(ibox, self, "grupo_var", model=self.grupo_model, searchable=True, callback=self.kfold_changed)
        caja_procesos(self)

    def _update_controls(self):
        super()._update_controls()
        self.grupo_model.set_domain(self.data.domain if self.data is not None else None)
        if self.grupo_var not in self.grupo_model:
            self.grupo_var = None

    def _OWTestAndScore__submit(self, testfunc):
        """Envuelve la función que armó el original antes de mandarla a correr."""
        self.Warning.sin_paralelo.clear()
        self.Warning.sin_grupo.clear()
        self.Warning.sin_agrupar.clear()
        try:
            testfunc = self._paralelizar(testfunc)
        except EstructuraDesconocida:
            self.Warning.sin_paralelo()
            # la CV agrupada es propia: el original corre la común, y los resultados no son comparables
            self.Warning.sin_agrupar(shown=self.resampling == self.KFold and self.cv_agrupada)
        owtestandscore.OWTestAndScore._OWTestAndScore__submit(self, testfunc)

    def _paralelizar(self, testfunc):
        """partial(replace_learners, partial(sampler, *args)) → la misma cadena con evaluar_muestreo adentro."""
        ok = (evaluacion.ORANGE_PROBADO and isinstance(testfunc, partial)
              and getattr(testfunc.func, "__name__", "") == "replace_learners"
              and len(testfunc.args) == 1 and not testfunc.keywords and isinstance(testfunc.args[0], partial)
              and isinstance(testfunc.args[0].func, Validation) and not testfunc.args[0].keywords)
        if not ok:
            raise EstructuraDesconocida
        sampler, args = testfunc.args[0].func, testfunc.args[0].args
        if self.resampling == self.KFold and self.cv_agrupada and type(sampler) is CrossValidation:
            if self.grupo_var is None:
                self.Warning.sin_grupo()
            else:
                sampler = evaluacion.CVAgrupada(k=sampler.k, grupo=self.grupo_var, stratified=sampler.stratified,
                                                random_state=sampler.random_state, store_data=sampler.store_data,
                                                store_models=sampler.store_models)
        return partial(testfunc.func, partial(evaluacion.evaluar_muestreo, sampler, args, self.n_procesos))

    def send_report(self):
        super().send_report()
        self.report_items("Parallel", [("Processes", self.n_procesos),
                                       ("Groups kept together",
                                        self.grupo_var.name if self.cv_agrupada and self.grupo_var else "—")])


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import LogisticRegressionLearner
    from Orange.data import Table
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWTestAndScoreParalelo).run(set_train_data=Table("heart_disease"),
                                             insert_learner=(0, LogisticRegressionLearner()))
