"""El motor: orden, progreso, serie, errores, cancelación y dos trabajos a la vez."""
import os
import threading
import time

import psutil
import pytest

from orangecontrib.paralelo import motor


def _cuadrado(x):
    return x * x


def _pid(_):
    return os.getpid()


def _falla_en_3(x):
    if x == 3:
        raise ValueError("tarea 3")
    time.sleep(0.05)
    return x


def _dormir(s):
    time.sleep(s)
    return os.getpid()


def _vivos(pids):
    return [p for p in pids if psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE]


def test_orden_y_progreso():
    visto = []
    r = motor.ejecutar(_cuadrado, [(i,) for i in range(20)], 3, progreso=lambda h, t: visto.append((h, t)))
    assert r == [i * i for i in range(20)]
    assert visto[-1] == (20, 20) and [h for h, _ in visto] == sorted(h for h, _ in visto)


def test_serie_en_este_proceso():
    assert set(motor.ejecutar(_pid, [(i,) for i in range(3)], 1)) == {os.getpid()}
    assert set(motor.ejecutar(_pid, [(0,)], 8)) == {os.getpid()}  # una sola tarea
    assert os.getpid() not in motor.ejecutar(_pid, [(i,) for i in range(4)], 2)


def test_error_de_una_tarea_se_propaga_y_no_deja_procesos():
    with pytest.raises(ValueError, match="tarea 3"):
        motor.ejecutar(_falla_en_3, [(i,) for i in range(8)], 2)
    assert motor._grupo is None
    assert motor.ejecutar(_cuadrado, [(2,), (3,)], 2) == [4, 9]  # el siguiente trabajo arranca limpio


def test_cancelar_mata_los_procesos():
    pids = motor.ejecutar(_pid, [(i,) for i in range(4)], 2)
    t0, pedido = time.time(), threading.Event()
    threading.Timer(0.5, pedido.set).start()
    with pytest.raises(motor.Cancelado):
        motor.ejecutar(_dormir, [(30,)] * 4, 2, cancelado=pedido.is_set)
    assert time.time() - t0 < 5
    time.sleep(1)
    assert not _vivos(set(pids))


def test_dos_trabajos_a_la_vez_y_se_cancela_el_que_espera():
    resultado, errores = {}, {}

    def largo():
        resultado["a"] = motor.ejecutar(_dormir, [(1.5,)] * 2, 2)

    def corto():
        try:
            motor.ejecutar(_cuadrado, [(1,), (2,)], 2, cancelado=lambda: time.time() - t0 > 0.5)
        except motor.Cancelado as e:
            errores["b"] = e

    t0 = time.time()
    a = threading.Thread(target=largo)
    a.start()
    time.sleep(0.2)  # a tiene el candado
    b = threading.Thread(target=corto)
    b.start()
    b.join(5)
    assert isinstance(errores.get("b"), motor.Cancelado) and a.is_alive()  # b se canceló esperando; a sigue
    a.join(10)
    assert len(resultado["a"]) == 2


def test_progreso_que_lanza_corta_el_trabajo():
    class Interrumpido(BaseException):
        pass

    def progreso(h, t):
        raise Interrumpido

    with pytest.raises(Interrumpido):
        motor.ejecutar(_dormir, [(0.1,)] * 6, 2, progreso=progreso)
    assert motor._grupo is None


def test_conviene_paralelo():
    assert not motor.conviene_paralelo(1, 10, 10**9)
    assert not motor.conviene_paralelo(8, 1, 10**9)
    assert not motor.conviene_paralelo(8, 10, motor.UMBRAL_TRABAJO - 1)
    assert motor.conviene_paralelo(8, 10, motor.UMBRAL_TRABAJO)
