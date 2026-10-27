"""
CLI Application Adapter
"""
import io
from PIL import Image, ImageDraw
from ....application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from ....application.dtos.estimate_dto import SolicitudEstimacionDTO
from ..detectors.mock_detector_adapter import MockFruitDetectorAdapter

def run_cli():
    print("==========================================================")
    print(" SIKAFE: Sistema Inteligente de Estimación de Cosecha")
    print(" Arquitectura Hexagonal (Ports & Adapters)")
    print("==========================================================")
    
    img = Image.new("RGB", (640, 480), color=(34, 139, 34))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 140, 140], fill=(220, 20, 60))
    draw.ellipse([200, 150, 240, 190], fill=(255, 215, 0))
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    
    detector_adapter = MockFruitDetectorAdapter(num_simulated_fruits=18)
    use_case = AnalizarImagenCosechaUseCase(detector_adapter=detector_adapter)
    
    solicitud = SolicitudEstimacionDTO(
        imagen_bytes=buf.getvalue(),
        peso_medio_fruto_g=1.8,
        factor_oclusion=1.25,
        plantas_muestra=1,
        plantas_totales_lote=2500,
        umbral_confianza=0.40
    )
    
    respuesta = use_case.ejecutar(solicitud)
    print(f"Total cerezas detectadas: {respuesta.total_frutos}")
    print(f"Conteo por estado: {respuesta.conteo_por_estado}")
    print(f"Porcentaje de Madurez: {respuesta.porcentaje_madurez}%")
    print(f"Rendimiento Estimado: {respuesta.rendimiento_estimado_kg} kg de café cereza")
    print("==========================================================")

if __name__ == "__main__":
    run_cli()
