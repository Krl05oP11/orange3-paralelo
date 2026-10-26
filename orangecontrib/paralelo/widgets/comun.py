"""Piezas de interfaz que comparten los widgets paralelos."""
import warnings

from AnyQt.QtCore import Qt

from Orange.version import short_version
from Orange.widgets import gui

from orangecontrib.paralelo import motor

# Orange 3.40 avisa que heredar de un widget «será desactivado en el futuro». Los cuatro widgets heredan a propósito (la
# guarda de versión está en evaluacion.VERSIONES_PROBADAS; README, «Risks»): solo se calla ese aviso, y solo para estas
# cuatro clases.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r"subclassing of widget classes is deprecated[\s\S]*"
                                r"(OWTestAndScore|OWParameterFitter|OWPermutationPlot|OWRandomForest) ")

# el aviso de los widgets cuando corren el original en serie (Orange no probado: evaluacion.ORANGE_PROBADO)
SIN_PARALELO = f"Orange {short_version} has not been tested with this add-on: running serially, like the original."


def caja_procesos(widget, attr="n_procesos", callback=None):
    """Caja «Parallel» con el número de procesos (1 = el camino original de Orange, en serie)."""
    box = gui.vBox(widget.controlArea, "Parallel")
    gui.spin(box, widget, attr, 1, motor.N_MAXIMO, label="Processes:", controlWidth=60, alignment=Qt.AlignRight,
             callback=callback,
             tooltip=f"Processes that share the work (1 = serial, like the original). This computer has "
                     f"{motor.N_DEFECTO + 1} physical cores. Small jobs run serially anyway.")
    return box
