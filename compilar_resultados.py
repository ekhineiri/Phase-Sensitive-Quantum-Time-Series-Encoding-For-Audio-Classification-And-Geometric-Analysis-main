
"""
compilar_resultados.py
----------------------
Recorre outputs/ y agrupa los resultados de cada ejecucion en un informe
legible: tabla comparativa de encodings primero, matrices kernel despues.

Uso:
    python compilar_resultados.py

Genera resumen_resultados.txt en la raiz del proyecto.
Borra este script cuando ya no lo necesites.
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = PROJECT_ROOT / "resumen_resultados.txt"

SEP  = "=" * 72
SEP2 = "-" * 72


# ── Parseo de archivos .txt ────────────────────────────────────────────────

def parse_txt(path):
    """Lee un *_kernel_matrix_*.txt y devuelve un dict con sus campos."""
    texto = path.read_text(encoding="utf-8")

    def campo(patron, tipo=str):
        m = re.search(patron, texto, flags=re.MULTILINE)
        return tipo(m.group(1).strip()) if m else None

    resultado = {
        "dataset":        campo(r"^dataset:\s*(.+)$", str),
        "encoding":       campo(r"^encoding:\s*(.+)$", str),
        "clases":         campo(r"^clases:\s*(.+)$", str),
        "intra_mean":     campo(r"intra_class_mean\s*:\s*([0-9.eE+\-]+)", float),
        "inter_mean":     campo(r"inter_class_mean\s*:\s*([0-9.eE+\-]+)", float),
        "ratio":          campo(r"intra_inter_ratio\s*:\s*([0-9.eE+\-]+)", float),
        "diag_mean":      campo(r"diag_mean\s*:\s*([0-9.eE+\-]+)", float),
        "accuracy":       campo(r"accuracy\s*:\s*([0-9.eE+\-]+)", float),
        "n_train":        campo(r"train\s*:\s*(\d+)\s*muestras", int),
        "n_test":         campo(r"test\s*:\s*(\d+)\s*muestras", int),
        "y_test":         campo(r"y_test\s*:\s*(\[.+?\])", str),
        "y_pred":         campo(r"y_pred\s*:\s*(\[.+?\])", str),
    }
    return resultado


def cargar_kernel(npy_path):
    """Carga la matriz kernel desde el .npy correspondiente."""
    if npy_path.exists():
        return np.load(npy_path)
    return None


# ── Formateo de tablas ─────────────────────────────────────────────────────

def tabla_comparativa(resultados_por_encoding):
    """Genera una tabla de texto comparando los encodings en columnas."""
    encodings = ["qtse", "ry", "phase"]
    presentes = [e for e in encodings if e in resultados_por_encoding]

    col = 14  # ancho de cada columna de encoding
    met = 25  # ancho de la columna de metrica

    def fila(nombre, valores, fmt="{:>14}"):
        celdas = "".join(fmt.format(v) for v in valores)
        return f"  {nombre:<{met}} {celdas}"

    linea_h = "  " + "-" * met + " " + ("  " + "-" * (col - 2)) * len(presentes)

    encabezado = fila("Metrica", [e.upper().center(col) for e in presentes], fmt="{:>14}")

    filas = [
        SEP2,
        encabezado,
        linea_h,
        fila("Intra-clase (media)",  [f"{resultados_por_encoding[e]['intra_mean']:.4f}" for e in presentes]),
        fila("Inter-clase (media)",  [f"{resultados_por_encoding[e]['inter_mean']:.6f}" for e in presentes]),
        fila("Ratio intra/inter",    [f"{resultados_por_encoding[e]['ratio']:.2f}" for e in presentes]),
        fila("Media diagonal",       [f"{resultados_por_encoding[e]['diag_mean']:.4f}" for e in presentes]),
        linea_h,
        fila("Accuracy SVM",         [f"{resultados_por_encoding[e]['accuracy']:.4f}" for e in presentes]),
        fila("Train / Test",         [f"{resultados_por_encoding[e]['n_train']}  /  {resultados_por_encoding[e]['n_test']}" for e in presentes]),
        fila("y_test",               [resultados_por_encoding[e]['y_test'] or "?" for e in presentes]),
        fila("y_pred",               [resultados_por_encoding[e]['y_pred'] or "?" for e in presentes]),
        SEP2,
    ]
    return "\n".join(filas)


def tabla_kernel(K, titulo):
    """Formatea una matriz numpy como tabla de texto con indices."""
    n = K.shape[0]
    ancho = 8
    cabecera = " " * 5 + "".join(f"{'s'+str(j):>{ancho}}" for j in range(n))
    lineas = [f"\n  {titulo}", cabecera]
    for i in range(n):
        fila = f"  s{i:<3}" + "".join(f"{K[i, j]:>{ancho}.4f}" for j in range(n))
        lineas.append(fila)
    return "\n".join(lineas)


# ── Agrupado y renderizado ─────────────────────────────────────────────────

def agrupar_por_ejecucion():
    """
    Devuelve un dict {(fecha, hora, dataset): {encoding: (datos_txt, npy_path)}}.
    """
    grupos = defaultdict(dict)
    for txt_path in sorted(OUTPUTS_DIR.rglob("*_kernel_matrix_*.txt")):
        partes_rel = txt_path.relative_to(OUTPUTS_DIR).parts
        if len(partes_rel) < 3:
            continue
        fecha, hora = partes_rel[0], partes_rel[1]
        datos = parse_txt(txt_path)
        if not datos["dataset"] or not datos["encoding"]:
            continue
        npy_path = txt_path.with_suffix(".npy")
        clave = (fecha, hora, datos["dataset"])
        grupos[clave][datos["encoding"]] = (datos, npy_path)
    return grupos


def renderizar_ejecucion(fecha, hora, dataset, encodings_dict):
    bloques = []

    # Cabecera de la ejecucion
    ejemplo = next(iter(encodings_dict.values()))[0]
    clases = ejemplo["clases"] or "?"
    bloques.append(
        f"\n{SEP}\n"
        f"  EJECUCION : {fecha}  {hora}\n"
        f"  Dataset   : {dataset}\n"
        f"  Clases    : {clases}\n"
        f"{SEP}"
    )

    # Tabla comparativa
    bloques.append("\nCOMPARATIVA DE ENCODINGS\n")
    resultados = {enc: datos for enc, (datos, _) in encodings_dict.items()}
    bloques.append(tabla_comparativa(resultados))

    # Matrices kernel individuales
    bloques.append("\n\nMATRICES KERNEL\n")
    for enc in ["qtse", "ry", "phase"]:
        if enc not in encodings_dict:
            continue
        _, npy_path = encodings_dict[enc]
        K = cargar_kernel(npy_path)
        if K is not None:
            bloques.append(tabla_kernel(K, f"Encoding: {enc.upper()}"))
        else:
            bloques.append(f"\n  Encoding: {enc.upper()}  (matriz no disponible)")

    return "\n".join(bloques)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    grupos = agrupar_por_ejecucion()

    if not grupos:
        print(f"No se encontraron resultados en {OUTPUTS_DIR}/")
        return

    secciones = []
    for (fecha, hora, dataset), encodings_dict in sorted(grupos.items()):
        secciones.append(renderizar_ejecucion(fecha, hora, dataset, encodings_dict))

    total = sum(len(v) for v in grupos.values())
    pie = f"\n\n{SEP}\nTotal de ejecuciones: {len(grupos)}   Total de resultados: {total}\n{SEP}\n"

    OUTPUT_FILE.write_text("\n".join(secciones) + pie, encoding="utf-8")
    print(f"Resumen generado: {OUTPUT_FILE}  ({len(grupos)} ejecucion(es), {total} resultados)")


if __name__ == "__main__":
    main()
