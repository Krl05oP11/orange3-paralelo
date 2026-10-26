"""Parameter Fitter (paralelo): el widget original con un proceso por punto de la grilla.

Hereda de OWParameterFitter y cambia solo la tarea que arranca (`start`): `run` del original pasa a `run_paralelo`,
que corre la original si el trabajo es chico.
"""
from functools import partial

from Orange.widgets import settings
from Orange.widgets.evaluate import owparameterfitter
from Orange.widgets.widget import Msg

from orangecontrib.paralelo import evaluacion, motor
from orangecontrib.paralelo.widgets.comun import SIN_PARALELO, caja_procesos


def run_paralelo(data, learner, fitted_parameter_props, initial_parameters, steps, state, *, n_procesos):
    trabajo = motor.celdas(data) * len(steps) * (1 + evaluacion.N_FOLD)
    if not motor.conviene_paralelo(n_procesos, len(steps), trabajo):
        return owparameterfitter.run(data, learner, fitted_parameter_props, initial_parameters, steps, state)
    state.set_status("Computing…")
    return evaluacion.busqueda(data, learner, fitted_parameter_props, initial_parameters, steps,
                               lambda hechas, total: state.set_progress_value(100 * hechas / total),
                               n_procesos=n_procesos, cancelado=state.is_interruption_requested)


class OWParameterFitterParalelo(owparameterfitter.OWParameterFitter):
    name = "Parameter Fitter (parallel)"
    description = "Parameter Fitter with the grid points spread across CPU cores."
    icon = "icons/ParameterFitter.svg"
    priority = 20
    keywords = "parameter fitter, grid, parallel, paralelo"
    replaces = []

    n_procesos = settings.Setting(motor.N_DEFECTO)

    class Warning(owparameterfitter.OWParameterFitter.Warning):
        sin_paralelo = Msg(SIN_PARALELO)

    def __init__(self):
        super().__init__()
        caja_procesos(self)

    def start(self, task, *args, **kwargs):
        self.Warning.sin_paralelo(shown=not evaluacion.ORANGE_PROBADO)
        if evaluacion.ORANGE_PROBADO and task is owparameterfitter.run:
            task = partial(run_paralelo, n_procesos=self.n_procesos)
        super().start(task, *args, **kwargs)


if __name__ == "__main__":  # pragma: no cover
    from Orange.data import Table
    from Orange.modelling import RandomForestLearner
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWParameterFitterParalelo).run(set_data=Table("heart_disease"), set_learner=RandomForestLearner())
