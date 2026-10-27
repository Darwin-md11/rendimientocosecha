"""
Primary Inbound Adapter (Gradio Web UI - Estilo Cafetero):
Interfaz web moderna, accesible y visualmente ambientada en el ecosistema cafetero.
Integrada bajo Arquitectura Hexagonal con el caso de uso AnalizarImagenCosechaUseCase.
"""
import io
import gradio as gr
from PIL import Image

from src.application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from src.application.dtos.estimate_dto import SolicitudEstimacionDTO
from src.infrastructure.adapters.detectors.mock_detector_adapter import MockFruitDetectorAdapter
from src.domain.exceptions.domain_exceptions import DomainException

# Inyección de dependencias (Puerto de Salida -> Adaptador Detector)
detector_adapter = MockFruitDetectorAdapter(num_simulated_fruits=22)
use_case = AnalizarImagenCosechaUseCase(detector_adapter=detector_adapter)

def procesar_estimacion(imagen, peso_g, oclusion, plantas_muestra, plantas_totales, umbral):
    if imagen is None:
        return None, """
        <div style='background-color: #FDEDEC; border-left: 5px solid #E74C3C; padding: 12px; border-radius: 6px; font-weight: bold; color: #78281F;'>
            ⚠️ Por favor cargue una fotografía RGB de cafeto para iniciar la estimación.
        </div>
        """, "", ""

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

        # Dashboard Visual de Resultados
        resumen_html = f"""
        <div style='background-color: #FDFEFE; padding: 20px; border-radius: 12px; border: 1px solid #E5E7E9; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
            <h3 style='margin-top: 0; color: #4A235A; border-bottom: 2px solid #D5F5E3; padding-bottom: 8px;'>🌱 Métricas Principales de Cosecha</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 15px;'>
                <div style='background: #F4ECF7; padding: 12px; border-radius: 8px; text-align: center;'>
                    <span style='font-size: 0.85em; color: #5B2C6F; font-weight: bold;'>TOTAL CEREZAS</span><br>
                    <span style='font-size: 1.8em; font-weight: bold; color: #4A235A;'>{respuesta.total_frutos}</span>
                </div>
                <div style='background: #E8F8F5; padding: 12px; border-radius: 8px; text-align: center;'>
                    <span style='font-size: 0.85em; color: #117A65; font-weight: bold;'>MADUREZ ÍNDICE</span><br>
                    <span style='font-size: 1.8em; font-weight: bold; color: #196F3D;'>{respuesta.porcentaje_madurez}%</span>
                </div>
                <div style='background: #FDEDEC; padding: 12px; border-radius: 8px; text-align: center;'>
                    <span style='font-size: 0.85em; color: #78281F; font-weight: bold;'>RENDIMIENTO ESTIMADO</span><br>
                    <span style='font-size: 1.8em; font-weight: bold; color: #922B21;'>{respuesta.rendimiento_estimado_kg} kg</span>
                </div>
            </div>
        </div>
        """

        conteo_md = f"""
        <div style='background-color: #FAF5EF; padding: 15px; border-radius: 10px; border-left: 5px solid #6E2C00; margin-top: 15px;'>
            <h4 style='margin-top:0; color: #4A235A;'>🍒 Clasificación por Estado de Madurez</h4>
            <ul style='list-style-type: none; padding-left: 0; font-size: 1.05em;'>
                <li style='margin-bottom: 6px;'>🟢 <b style='color: #1E8449;'>Verde (Inmaduro):</b> {respuesta.conteo_por_estado.get('VERDE', 0)} frutos</li>
                <li style='margin-bottom: 6px;'>🟡 <b style='color: #D4AC0D;'>Pintón (En Transición):</b> {respuesta.conteo_por_estado.get('PINTON', 0)} frutos</li>
                <li style='margin-bottom: 6px;'>🔴 <b style='color: #C0392B;'>Maduro (Óptimo Recolección):</b> {respuesta.conteo_por_estado.get('MADURO', 0)} frutos</li>
                <li style='margin-bottom: 6px;'>🟣 <b style='color: #7D3C98;'>Sobremaduro (Seco/Pasado):</b> {respuesta.conteo_por_estado.get('SOBREMADURO', 0)} frutos</li>
            </ul>
        </div>
        """

        status_html = """
        <div style='background-color: #D5F5E3; color: #196F3D; padding: 10px; border-radius: 6px; font-weight: bold;'>
            ✅ Análisis visual y cálculo de rendimiento completados correctamente.
        </div>
        """

        return imagen_anotada, resumen_html, conteo_md, status_html

    except DomainException as e:
        err_html = f"<div style='background-color: #FDEDEC; color: #78281F; padding: 10px; border-radius: 6px;'>❌ <b>Error de Dominio:</b> {str(e)}</div>"
        return None, err_html, "", ""
    except Exception as e:
        err_html = f"<div style='background-color: #FDEDEC; color: #78281F; padding: 10px; border-radius: 6px;'>❌ <b>Error Inesperado:</b> {str(e)}</div>"
        return None, err_html, "", ""

# CSS de Personalización Temática Cafetera
custom_css = """
body, .gradio-container {
    background-color: #F5EBE6 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header-box {
    background: linear-gradient(135deg, #3E2723 0%, #4E342E 50%, #6D4C41 100%);
    color: #FFFFFF;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.header-box h1 {
    color: #FFECB3 !important;
    margin-bottom: 5px;
}
.header-box p {
    color: #D7CCC8 !important;
    font-size: 1.1em;
}
.btn-coffee {
    background-color: #6E2C00 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: none !important;
}
.btn-coffee:hover {
    background-color: #4A1E00 !important;
}
"""

with gr.Blocks(title="SIKAFE - Estimación Cosecha Café", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div class='header-box'>
            <h1>☕ SIKAFE: Sistema Inteligente de Estimación de Cosecha de Café</h1>
            <p>🌿 <i>Agronomía Digital y Visión por Computador para la Caficultura de Norte de Santander</i></p>
            <span style='background-color: #D7CCC8; color: #3E2723; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.85em;'>
                Arquitectura Hexagonal (Ports & Adapters)
            </span>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Fotografía de Cafeto y Parámetros del Lote")
            input_image = gr.Image(type="pil", label="Cargar Fotografía RGB de la Rama / Árbol")
            
            with gr.Accordion("⚙️ Parámetros Agronómicos de la Finca", open=True):
                peso_g = gr.Number(value=1.8, label="Peso Medio Cereza (g)", info="Estándar Variedad Castillo / Bourbon (~1.8g)")
                oclusion = gr.Number(value=1.25, label="Factor Oclusión Foliares (≥ 1.0)", info="Compensación de frutos ocultos por hojas")
                plantas_muestra = gr.Number(value=1, label="N° Plantas en la Fotografía", precision=0)
                plantas_totales = gr.Number(value=2500, label="Plantas Totales en la Parcela / Lote", precision=0)
                umbral = gr.Slider(minimum=0.1, maximum=1.0, value=0.40, step=0.05, label="Umbral de Confianza de Detección (IA)")

            btn_calcular = gr.Button("☕ Calcular Rendimiento de Cosecha", elem_classes=["btn-coffee"])

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Inferencia IA y Reporte Agronómico")
            output_image = gr.Image(type="pil", label="Cerezas Detectadas y Clasificadas")
            output_status = gr.HTML()
            output_resumen = gr.HTML()
            output_conteo = gr.HTML()

    btn_calcular.click(
        fn=procesar_estimacion,
        inputs=[input_image, peso_g, oclusion, plantas_muestra, plantas_totales, umbral],
        outputs=[output_image, output_resumen, output_conteo, output_status]
    )

if __name__ == "__main__":
    demo.launch()