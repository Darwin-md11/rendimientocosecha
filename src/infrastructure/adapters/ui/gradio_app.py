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
import pandas as pd
import gradio as gr
from PIL import Image

from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from src.application.dtos.estimate_dto import SolicitudEstimacionDTO
from src.infrastructure.adapters.detectors.mock_detector_adapter import MockFruitDetectorAdapter
from src.domain.exceptions.domain_exceptions import DomainException

# Inyección de dependencias (Puerto de Salida -> Adaptador Detector)
detector_adapter = MockFruitDetectorAdapter(num_simulated_fruits=24)
use_case = AnalizarImagenCosechaUseCase(detector_adapter=detector_adapter)

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
        imagen.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        solicitud = SolicitudEstimacionDTO(
            imagen_bytes=img_bytes,
            peso_medio_fruto_g=float(peso_g),
            factor_oclusion=float(oclusion),
            plantas_muestra=int(plantas_muestra),
            plantas_totales_lote=int(plantas_totales),
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
            "✅ Inferencia y estimación agronómica completadas.",
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