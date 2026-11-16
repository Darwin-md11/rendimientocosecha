"""
Primary Inbound Adapter (Gradio Enterprise UI - Opción 1):
Interfaz limpia, minimalista y profesional utilizando componentes nativos de Gradio.
Consume el puerto de entrada AnalizarImagenCosechaUseCase respetando la Arquitectura Hexagonal.
"""
import sys
from pathlib import Path

# Inyección del sys.path para evitar ModuleNotFoundError
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import io

import gradio as gr
import pandas as pd
from PIL import Image

from src.application.dtos.estimate_dto import SolicitudEstimacionDTO
from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from src.domain.exceptions.domain_exceptions import DomainException
from src.infrastructure.adapters.ui.presentacion import banner_markdown, mensaje_estado
from src.infrastructure.composition import construir_detector

# Inyección de dependencias (Puerto de Salida -> Adaptador Detector).
# Ver composition.py: si el modelo configurado no carga, falla aqui con un mensaje claro.
detector_adapter = construir_detector()
use_case = AnalizarImagenCosechaUseCase(detector_adapter=detector_adapter)


def _numero(valor, nombre, por_defecto=None):
    """gr.Number devuelve None si el usuario borra el campo."""
    if valor is None:
        if por_defecto is None:
            raise DomainException(f"El campo '{nombre}' no puede estar vacío.")
        return por_defecto
    return valor


def procesar_estimacion(imagen, peso_g, oclusion, plantas_muestra, plantas_totales, umbral):
    if imagen is None:
        return (
            None,
            "⚠️ Ingrese una fotografía RGB válida para continuar.",
            0,
            0.0,
            0.0,
            pd.DataFrame(columns=["Estado", "Cantidad", "Proporción (%)"])
        )

    try:
        buffer = io.BytesIO()
        # convert("RGB"): las PNG/webcam pueden venir en RGBA y JPEG no admite canal alfa.
        imagen.convert("RGB").save(buffer, format="JPEG", quality=95)
        img_bytes = buffer.getvalue()

        solicitud = SolicitudEstimacionDTO(
            imagen_bytes=img_bytes,
            peso_medio_fruto_g=float(_numero(peso_g, "Peso Medio")),
            factor_oclusion=float(_numero(oclusion, "Factor Oclusión")),
            plantas_muestra=int(_numero(plantas_muestra, "Plantas Muestra")),
            plantas_totales_lote=int(_numero(plantas_totales, "Plantas Lote Total")),
            umbral_confianza=float(umbral)
        )

        respuesta = use_case.ejecutar(solicitud)
        imagen_anotada = Image.open(io.BytesIO(respuesta.imagen_anotada_bytes))

        # Construcción de la tabla limpia de madurez
        conteo = respuesta.conteo_por_estado
        total = respuesta.total_frutos if respuesta.total_frutos > 0 else 1

        df_datos = [
            {"Estado": "🟢 Verde (Inmaduro)", "Cantidad": conteo.get('VERDE', 0), "Proporción (%)": round((conteo.get('VERDE', 0)/total)*100, 1)},
            {"Estado": "🟡 Pintón (Transición)", "Cantidad": conteo.get('PINTON', 0), "Proporción (%)": round((conteo.get('PINTON', 0)/total)*100, 1)},
            {"Estado": "🔴 Maduro (Óptimo)", "Cantidad": conteo.get('MADURO', 0), "Proporción (%)": round((conteo.get('MADURO', 0)/total)*100, 1)},
            {"Estado": "🟣 Sobremaduro (Seco)", "Cantidad": conteo.get('SOBREMADURO', 0), "Proporción (%)": round((conteo.get('SOBREMADURO', 0)/total)*100, 1)}
        ]
        df_madurez = pd.DataFrame(df_datos)

        return (
            imagen_anotada,
            mensaje_estado(respuesta),
            respuesta.total_frutos,
            respuesta.porcentaje_madurez,
            respuesta.rendimiento_estimado_kg,
            df_madurez
        )

    except DomainException as e:
        return None, f"❌ Error de Dominio: {str(e)}", 0, 0.0, 0.0, pd.DataFrame()
    except Exception as e:
        return None, f"❌ Error de Sistema: {str(e)}", 0, 0.0, 0.0, pd.DataFrame()

# Configuración del Tema Limpio compatible con todas las versiones de Gradio
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.slate,
)

with gr.Blocks(title="SIKAFE - Dashboard Agronómico", theme=theme) as demo:
    gr.Markdown(
        """
        # SIKAFE — Sistema Inteligente de Estimación de Cosecha de Café
        *Plataforma de Visión Artificial para la Estimación de Rendimiento Agronómico (Norte de Santander)*
        """
    )
    # Banner permanente: procedencia del modelo y advertencia si no está validado.
    gr.Markdown(banner_markdown(detector_adapter.info()))

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("#### 📥 Captura de Fotografía y Parámetros del Lote")
            input_image = gr.Image(type="pil", label="Fotografía RGB del Cafeto", sources=["upload", "webcam"])

            with gr.Group():
                gr.Markdown("##### ⚙️ Parámetros Agronómicos")
                with gr.Row():
                    peso_g = gr.Number(value=1.8, label="Peso Medio (g)", info="Por cereza")
                    oclusion = gr.Number(value=1.25, label="Factor Oclusión", info="≥ 1.0 (Sombras/Follaje)")
                with gr.Row():
                    plantas_muestra = gr.Number(value=1, label="Plantas Muestra", precision=0)
                    plantas_totales = gr.Number(value=2500, label="Plantas Lote Total", precision=0)

                umbral = gr.Slider(minimum=0.1, maximum=1.0, value=0.40, step=0.05, label="Umbral de Confianza IA")

            btn_calcular = gr.Button("Analizar Fotografía y Estimar Rendimiento", variant="primary", size="lg")

        with gr.Column(scale=7):
            gr.Markdown("#### 📊 Reporte Agronómico e Inferencia")
            output_status = gr.Textbox(label="Estado del Procesamiento", interactive=False)

            with gr.Row():
                kpi_total = gr.Number(label="Total Cerezas", interactive=False)
                kpi_madurez = gr.Number(label="Índice Madurez (%)", interactive=False)
                kpi_yield = gr.Number(label="Rendimiento (kg)", interactive=False)

            output_image = gr.Image(type="pil", label="Detección de Objetos (Bounding Boxes)")
            output_df = gr.Dataframe(label="Desglose por Estado de Maduración", interactive=False)

    btn_calcular.click(
        fn=procesar_estimacion,
        inputs=[input_image, peso_g, oclusion, plantas_muestra, plantas_totales, umbral],
        outputs=[output_image, output_status, kpi_total, kpi_madurez, kpi_yield, output_df]
    )

if __name__ == "__main__":
    demo.launch()
