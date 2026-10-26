"""SVM lineal (liblinear): un modelo nuevo, no una copia paralela de uno de Orange (`svm_lineal.SVMLinealLearner`).

Pensado para texto (Bag of Words): ~1 s por pliegue donde el SVM de Orange tarda horas, misma clase predicha que el
argmax de sus probabilidades y resultados que no cambian de una corrida a otra. Sirve en cualquier Test and Score.
"""
from collections import OrderedDict

from AnyQt.QtCore import Qt

from Orange.widgets import gui, settings
from Orange.widgets.utils import owlearnerwidget

from orangecontrib.paralelo.svm_lineal import N_CALIBRACION, SVMLinealLearner


class OWSVMLineal(owlearnerwidget.OWBaseLearner):
    name = "Linear SVM (liblinear)"
    description = ("Linear SVM with liblinear, one-vs-one, with the data scaled like Orange's SVMs: fast on text, "
                   "deterministic, and with calibrated probabilities that agree with the predicted class.")
    icon = "icons/SVM.svg"
    priority = 50
    keywords = "svm, linear svm, svm lineal, liblinear, linearsvc, text, texto"
    LEARNER = SVMLinealLearner

    C = settings.Setting(1.0)
    balanceadas = settings.Setting(False)
    max_iter = settings.Setting(5000)

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, "Parameters")
        gui.doubleSpin(box, self, "C", 0.001, 1000.0, 0.1, decimals=3, label="Cost (C):", alignment=Qt.AlignRight,
                       controlWidth=80, callback=self.settings_changed,
                       tooltip="Higher: fits the training data more closely. Lower: more regularization.")
        gui.checkBox(box, self, "balanceadas", "Balanced classes", callback=self.settings_changed,
                     tooltip="Weights each class by its frequency (class_weight='balanced'). Classification only.")
        gui.spin(box, self, "max_iter", 100, 1_000_000, step=100, label="Maximum iterations:", alignment=Qt.AlignRight,
                 controlWidth=80, callback=self.settings_changed)
        gui.widgetLabel(box, f"Probabilities: calibrated sigmoid\n({N_CALIBRACION} internal folds, no shuffling).")

    def create_learner(self):
        return self.LEARNER(C=self.C, class_weight="balanced" if self.balanceadas else None, max_iter=self.max_iter,
                            preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        return OrderedDict([("Cost (C)", self.C), ("Balanced classes", "yes" if self.balanceadas else "no"),
                            ("Maximum iterations", self.max_iter),
                            ("Probabilities", f"calibrated sigmoid, {N_CALIBRACION} folds")])


if __name__ == "__main__":  # pragma: no cover
    from Orange.data import Table
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSVMLineal).run(Table("iris"))
