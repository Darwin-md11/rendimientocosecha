"""
Adaptador de entrada: interfaz web (Gradio).
Consume el caso de uso AnalizarImagenCosechaUseCase (arquitectura hexagonal); los fragmentos visuales
salen de presentacion.py, que no depende de Gradio y está probado.

Unidad de muestreo (decisión del equipo, ADR-011): cada foto muestra UNA planta completa.
"""
import inspect
import io
import os
import sys
from pathlib import Path

# Permite ejecutar este archivo directamente (python src/.../gradio_app.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.application.dtos.estimate_dto import SolicitudEstimacionDTO  # noqa: E402
from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase  # noqa: E402
from src.domain.exceptions.domain_exceptions import DomainException  # noqa: E402
from src.infrastructure.adapters.ui import presentacion as P  # noqa: E402
from src.infrastructure.composition import construir_catalogo, construir_detector  # noqa: E402

# Inyección de dependencias (composition.py): si el modelo configurado no carga, falla aquí con un mensaje claro.
detector_adapter = construir_detector()
catalogo = construir_catalogo()  # valores por defecto y fuentes: configs/agronomia.yaml
use_case = AnalizarImagenCosechaUseCase(detector_adapter=detector_adapter, catalogo=catalogo)
info_modelo = detector_adapter.info()

COLUMNAS = ["Estado", "Cerezas", "% del total", "Qué significa"]


def _numero(valor, nombre):
    """gr.Number devuelve None si el usuario borra el campo."""
    if valor is None:
        raise DomainException(f"El campo '{nombre}' no puede estar vacío.")
    return valor


def _tabla(respuesta) -> pd.DataFrame:
    filas = [(n, c, P.numero(p, 1), d) for _, n, d, c, p in P.filas_madurez(respuesta)]
    filas.append(("Total", respuesta.total_frutos, "100" if respuesta.total_frutos else "0", ""))
    return pd.DataFrame(filas, columns=COLUMNAS)


def procesar_estimacion(imagen, plantas_totales, peso_g, oclusion, umbral):
    """Devuelve (imagen anotada, resumen HTML, tabla de madurez, cálculo HTML)."""
    vacia = pd.DataFrame(columns=COLUMNAS)
    if imagen is None:
        return None, P.error_html("Suba una fotografía de una planta completa."), vacia, ""
    try:
        buffer = io.BytesIO()
        # convert("RGB"): las PNG/webcam pueden venir en RGBA y JPEG no admite canal alfa.
        imagen.convert("RGB").save(buffer, format="JPEG", quality=95)
        solicitud = SolicitudEstimacionDTO(
            imagen_bytes=buffer.getvalue(),
            peso_medio_fruto_g=float(_numero(peso_g, "Peso medio de una cereza")),
            factor_oclusion=float(_numero(oclusion, "Factor de oclusión")),
            plantas_muestra=None,  # una foto = una planta (configs/agronomia.yaml)
            plantas_totales_lote=int(_numero(plantas_totales, "Plantas en el lote")),
            umbral_confianza=float(umbral),
        )
        r = use_case.ejecutar(solicitud)
        resumen = P.tarjetas_html(r) + P.madurez_html(r) + P.avisos_html(r)
        return Image.open(io.BytesIO(r.imagen_anotada_bytes)), resumen, _tabla(r), P.calculo_html(r)
    except DomainException as e:
        return None, P.error_html(str(e)), vacia, ""
    except Exception as e:  # noqa: BLE001 - se muestra al usuario en vez de romper la interfaz
        return None, P.error_html(f"Error del sistema: {e}"), vacia, ""


def _fuente(clave: str) -> str:
    return catalogo.factores[clave].fuente_mostrada


theme = gr.themes.Soft(primary_hue=gr.themes.colors.emerald, secondary_hue=gr.themes.colors.neutral, neutral_hue=gr.themes.colors.slate)
# Gradio 4-5 reciben tema y CSS en Blocks(); Gradio 6, en launch().
_ESTILO = {"theme": theme, "css": P.CSS}
_EN_BLOCKS = "css" in inspect.signature(gr.Blocks.__init__).parameters

with gr.Blocks(title="SIKAFE · Estimación de cosecha de café", **(_ESTILO if _EN_BLOCKS else {})) as demo:
    gr.HTML(P.encabezado_html(info_modelo))

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            input_image = gr.Image(type="pil", label="Foto de una planta completa", sources=["upload", "webcam"], height=340)
            plantas_totales = gr.Number(
                value=catalogo.valor("plantas_totales_lote"), label="Plantas en el lote", precision=0, minimum=1,
                info="Lleva el resultado de la planta fotografiada a todo el lote.",
            )
            with gr.Accordion("Parámetros avanzados", open=False):
                peso_g = gr.Number(
                    value=catalogo.valor("peso_medio_fruto_g"), label="Peso medio de una cereza (g)", minimum=0.01,
                    info=f"Fuente: {_fuente('peso_medio_fruto_g')}",
                )
                oclusion = gr.Number(
                    value=catalogo.valor("factor_oclusion"), label="Factor de oclusión", minimum=1.0,
                    info="Cerezas reales por cada cereza visible (hojas, otra cara de la planta). 1,0 = sin corrección; pendiente de calibrar.",
                )
                umbral = gr.Slider(
                    minimum=0.05, maximum=0.95, value=P.UMBRAL_POR_DEFECTO, step=0.05, label="Umbral de confianza del detector",
                    info="0,25 es el valor con que se evaluó el modelo. Más alto: menos detecciones falsas, pero se pierden cerezas.",
                )
            btn_calcular = gr.Button("Analizar foto", variant="primary", size="lg")
            with gr.Accordion("Procedencia del modelo", open=False):
                gr.HTML(P.procedencia_html(info_modelo))

        with gr.Column(scale=6, min_width=420):
            resumen = gr.HTML(P.vacio_html())
            with gr.Tabs():
                with gr.Tab("Foto analizada"):
                    output_image = gr.Image(type="pil", label="Cerezas detectadas", interactive=False, height=440)
                with gr.Tab("Desglose por madurez"):
                    output_df = gr.Dataframe(value=pd.DataFrame(columns=COLUMNAS), interactive=False, wrap=True)
                with gr.Tab("Cómo se calculó"):
                    calculo = gr.HTML()

    btn_calcular.click(
        fn=procesar_estimacion,
        inputs=[input_image, plantas_totales, peso_g, oclusion, umbral],
        outputs=[output_image, resumen, output_df, calculo],
    )

if __name__ == "__main__":
    # SIKAFE_COMPARTIR=1 crea un enlace público temporal de Gradio (demo en Colab). Por defecto, solo local.
    demo.launch(share=os.environ.get("SIKAFE_COMPARTIR") == "1", **({} if _EN_BLOCKS else _ESTILO))
