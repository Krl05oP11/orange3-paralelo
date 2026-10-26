"""Random Forest (paralelo): el widget original con los árboles repartidos entre núcleos (n_jobs de scikit-learn).

Dentro de un widget paralelo (Test and Score, Parameter Fitter, Permutation Plot) cada proceso lo usa con n_jobs = 1:
los núcleos ya se reparten afuera (`evaluacion._un_hilo`).
"""
from Orange.widgets import settings
from Orange.widgets.model import owrandomforest

from orangecontrib.paralelo import motor
from orangecontrib.paralelo.widgets.comun import caja_procesos


class OWRandomForestParalelo(owrandomforest.OWRandomForest):
    name = "Random Forest (parallel)"
    description = "Random Forest that trains its trees in parallel."
    icon = "icons/RandomForest.svg"
    priority = 40
    keywords = "random forest, parallel, paralelo"
    replaces = []
    LEARNER = owrandomforest.OWRandomForest.LEARNER  # el metaclass de los widgets de modelos lo pide en cada clase

    n_procesos = settings.Setting(motor.N_DEFECTO)

    def add_main_layout(self):
        super().add_main_layout()
        caja_procesos(self, callback=self.settings_changed)

    def create_learner(self):
        learner = super().create_learner()
        learner.kwargs["n_jobs"] = self.n_procesos  # recién creado: sus aprendices todavía no existen
        return learner


if __name__ == "__main__":  # pragma: no cover
    from Orange.data import Table
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWRandomForestParalelo).run(Table("iris"))
