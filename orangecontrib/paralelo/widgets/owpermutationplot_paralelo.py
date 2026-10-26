"""Permutation Plot (paralelo): el widget original con un proceso por permutación.

Hereda de OWPermutationPlot y cambia solo la tarea que arranca (`start`). Los barajes se generan antes, en el orden y con
la semilla del original: el resultado es el mismo si el modelo no usa el generador global de numpy (ver README).
"""
from functools import partial

from Orange.widgets import settings
from Orange.widgets.evaluate import owpermutationplot
from Orange.widgets.widget import Msg

from orangecontrib.paralelo import evaluacion, motor
from orangecontrib.paralelo.widgets.comun import SIN_PARALELO, caja_procesos


def run_paralelo(data, learner, n_perm, state, *, n_procesos):
    trabajo = motor.celdas(data) * (n_perm + 1) * (1 + evaluacion.N_FOLD)
    if not motor.conviene_paralelo(n_procesos, n_perm + 1, trabajo):
        return owpermutationplot.run(data, learner, n_perm, state)
    state.set_status("Computing…")
    return evaluacion.permutaciones(data, learner, n_perm,
                                    lambda hechas, total: state.set_progress_value(100 * hechas / total),
                                    n_procesos=n_procesos, cancelado=state.is_interruption_requested)


class OWPermutationPlotParalelo(owpermutationplot.OWPermutationPlot):
    name = "Permutation Plot (parallel)"
    description = "Permutation Plot with the permutations spread across CPU cores."
    icon = "icons/PermutationPlot.svg"
    priority = 30
    keywords = "permutation plot, parallel, paralelo"
    replaces = []

    n_procesos = settings.Setting(motor.N_DEFECTO)

    class Warning(owpermutationplot.OWPermutationPlot.Warning):
        sin_paralelo = Msg(SIN_PARALELO)

    def __init__(self):
        super().__init__()
        caja_procesos(self)

    def start(self, task, *args, **kwargs):
        self.Warning.sin_paralelo(shown=not evaluacion.ORANGE_PROBADO)
        if evaluacion.ORANGE_PROBADO and task is owpermutationplot.run:
            task = partial(run_paralelo, n_procesos=self.n_procesos)
        super().start(task, *args, **kwargs)


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import LogisticRegressionLearner
    from Orange.data import Table
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWPermutationPlotParalelo).run(set_data=Table("iris"), set_learner=LogisticRegressionLearner())
