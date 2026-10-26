import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # los módulos de widgets importan Qt


@pytest.fixture(autouse=True, scope="session")
def _apagar_motor():
    yield
    from orangecontrib.paralelo import motor
    motor.apagar()
