"""Motor común: reparte tareas independientes entre procesos, con 1 hilo de BLAS por proceso.

- Un solo grupo de procesos (loky) para todo Orange, de tamaño fijo mientras corre un trabajo. Un candado deja correr un
  trabajo paralelo a la vez: si dos widgets arrancan juntos, el segundo espera (y se puede cancelar mientras espera).
- Cancelar o fallar mata los procesos del grupo (solo los usa ese trabajo) y el próximo trabajo arranca uno nuevo.
- Los datos grandes se guardan una vez en disco (`compartido`) y cada proceso los abre mapeados (`cargar`), en lugar de
  copiarlos en cada tarea.
- Este módulo no importa Qt: los procesos solo cargan lo que necesitan las tareas.
"""
import os
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import contextmanager

import joblib
import psutil
from joblib.externals.loky.process_executor import ProcessPoolExecutor

N_DEFECTO = max(1, (psutil.cpu_count(logical=False) or 2) - 1)  # núcleos físicos − 1: la interfaz sigue respondiendo
N_MAXIMO = os.cpu_count() or 1
# Por debajo de esto (celdas × ajustes de modelos) corre en serie: arrancar los procesos (~1 s, el primer trabajo de
# la sesión y después de cancelar) cuesta más que el trabajo. Calibrado midiendo los dos caminos: en frío,
# serie y paralelo empatan cerca de 1,2 millones (1,27 s frente a 1,18 s); en caliente el paralelo gana siempre.
UMBRAL_TRABAJO = 1_000_000
_UN_HILO = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
_ESPERA = 0.2  # segundos entre consultas de cancelación

_candado = threading.Lock()
_grupo = None


class Cancelado(Exception):
    """El widget pidió cancelar: los procesos se mataron."""


def celdas(data):
    """Tamaño de una tabla para decidir serie o paralelo: celdas no nulas si X es dispersa, filas × columnas si no."""
    X = data.X
    return X.nnz if hasattr(X, "nnz") else X.size


def conviene_paralelo(n_procesos, n_tareas, trabajo):
    """Decide antes de repartir (nunca corre una tarea de prueba): paralelo si hay procesos, tareas y trabajo."""
    return n_procesos > 1 and n_tareas > 1 and trabajo >= UMBRAL_TRABAJO


def _matar():
    global _grupo
    if _grupo is not None:
        _grupo.shutdown(wait=False, kill_workers=True)
        _grupo = None


def _grupo_de(n):
    """El grupo de procesos, con al menos n. Solo se llama con el candado tomado: ningún trabajo lo está usando."""
    global _grupo
    if _grupo is None or _grupo._max_workers < n:
        if _grupo is not None:
            _grupo.shutdown(wait=True)
        _grupo = ProcessPoolExecutor(max_workers=n, env=_UN_HILO)
    return _grupo


def ejecutar(funcion, tareas, n_procesos, progreso=None, cancelado=None):
    """[funcion(*t) for t in tareas], repartido entre n_procesos; los resultados vuelven en el orden de las tareas.

    funcion tiene que estar definida a nivel de módulo, en un módulo sin Qt. progreso(hechas, total) corre en este
    proceso a medida que terminan las tareas (si lanza una excepción, el trabajo se corta como si se cancelara);
    cancelado() se consulta cada 0,2 s. Una excepción de una tarea se propaga tal cual. Con n_procesos ≤ 1 o una sola
    tarea corre en serie, en este proceso.
    """
    tareas = list(tareas)
    total = len(tareas)
    progreso = progreso or (lambda hechas, total: None)
    cancelado = cancelado or (lambda: False)
    if n_procesos <= 1 or total <= 1:
        resultados = []
        for t in tareas:
            if cancelado():
                raise Cancelado
            resultados.append(funcion(*t))
            progreso(len(resultados), total)
        return resultados
    n = min(n_procesos, total, N_MAXIMO)
    while not _candado.acquire(timeout=_ESPERA):
        if cancelado():
            raise Cancelado
    try:
        grupo = _grupo_de(n)
        resultados, pendientes, siguiente, hechas = [None] * total, {}, 0, 0
        try:
            while hechas < total:
                while siguiente < total and len(pendientes) < n:  # como mucho n tareas a la vez
                    pendientes[grupo.submit(funcion, *tareas[siguiente])] = siguiente
                    siguiente += 1
                listas, _ = wait(pendientes, timeout=_ESPERA, return_when=FIRST_COMPLETED)
                if cancelado():
                    raise Cancelado
                for f in listas:
                    resultados[pendientes.pop(f)] = f.result()
                    hechas += 1
                if listas:
                    progreso(hechas, total)
        except BaseException:
            _matar()  # cancelación, error de una tarea o del callback: nada queda corriendo
            raise
        return resultados
    finally:
        _candado.release()


@contextmanager
def compartido(obj):
    """Guarda obj (p. ej. una Table) una vez en un directorio temporal y da su ruta; se borra al salir."""
    with tempfile.TemporaryDirectory(prefix="orange-paralelo-") as d:
        ruta = os.path.join(d, "datos.pkl")
        joblib.dump(obj, ruta)
        yield ruta


_cargado = {}


def cargar(ruta):
    """En un proceso de trabajo: el objeto de `compartido`, mapeado desde el disco y cargado una sola vez por trabajo."""
    if ruta not in _cargado:
        _cargado.clear()
        _cargado[ruta] = joblib.load(ruta, mmap_mode="r")
    return _cargado[ruta]


def apagar():
    """Termina el grupo de procesos (al cerrar Orange o en las pruebas)."""
    global _grupo
    with _candado:
        if _grupo is not None:
            _grupo.shutdown(wait=True)
            _grupo = None
