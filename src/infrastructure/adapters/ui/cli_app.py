"""
CLI Application Adapter

Ejemplos:
    python -m src.infrastructure.adapters.ui.cli_app                      # escena sintetica de demostracion
    python -m src.infrastructure.adapters.ui.cli_app --imagen foto.jpg    # foto real
    python -m src.infrastructure.adapters.ui.cli_app --detector mock      # detector simulado

Siempre imprime la procedencia del modelo y, si no esta validado con datos
reales, una advertencia visible.
"""
import argparse
from pathlib import Path

from ....application.dtos.estimate_dto import SolicitudEstimacionDTO
from ....application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from ....domain.exceptions.domain_exceptions import DomainException
from ...composition import DETECTORES_DISPONIBLES, construir_catalogo, construir_detector
from .presentacion import UMBRAL_POR_DEFECTO, lineas_cli

IMAGEN_DEMO = Path(__file__).resolve().parents[4] / "assets" / "demo_sintetica.jpg"


def run_cli(argv=None) -> int:
    p = argparse.ArgumentParser(description="SIKAFE: estimacion de cosecha de cafe.")
    p.add_argument("--imagen", type=Path, help="foto del cafeto (si se omite, usa una escena sintetica)")
    p.add_argument("--detector", choices=[*DETECTORES_DISPONIBLES, "ml"], default=None)
    p.add_argument("--peso-g", type=float, help="peso medio por cereza en g (por defecto: configs/agronomia.yaml)")
    p.add_argument("--oclusion", type=float, help="factor de oclusion >= 1.0 (por defecto: configs/agronomia.yaml)")
    p.add_argument("--plantas-muestra", type=int, help="plantas que cubre la foto (por defecto 1: una foto = una planta completa)")
    p.add_argument("--plantas-lote", type=int, help="plantas del lote (por defecto: configs/agronomia.yaml)")
    p.add_argument("--umbral", type=float, default=UMBRAL_POR_DEFECTO, help="umbral de confianza (0 a 1; 0,25 = el de la evaluación)")
    p.add_argument("--guardar-anotada", type=Path, help="ruta donde guardar la imagen con las detecciones")
    args = p.parse_args(argv)

    print("==========================================================")
    print(" SIKAFE: Sistema Inteligente de Estimación de Cosecha")
    print(" Arquitectura Hexagonal (Ports & Adapters)")
    print("==========================================================")

    if args.imagen:
        if not args.imagen.is_file():
            print(f"No existe la imagen: {args.imagen}")
            return 1
        imagen_bytes = args.imagen.read_bytes()
    else:
        print("(Sin --imagen: se usa una escena sintética de demostración)")
        imagen_bytes = IMAGEN_DEMO.read_bytes()

    try:
        use_case = AnalizarImagenCosechaUseCase(detector_adapter=construir_detector(args.detector), catalogo=construir_catalogo())
    except DomainException as e:
        print(f"Error al cargar el detector: {e}")
        return 1
    solicitud = SolicitudEstimacionDTO(
        imagen_bytes=imagen_bytes,
        peso_medio_fruto_g=args.peso_g,
        factor_oclusion=args.oclusion,
        plantas_muestra=args.plantas_muestra,
        plantas_totales_lote=args.plantas_lote,
        umbral_confianza=args.umbral,
    )

    try:
        respuesta = use_case.ejecutar(solicitud)
    except DomainException as e:
        print(f"Error: {e}")
        return 1

    print("\n".join(lineas_cli(respuesta)))
    print(f"Total cerezas detectadas: {respuesta.total_frutos}")
    print(f"Conteo por estado: {respuesta.conteo_por_estado}")
    print(f"Porcentaje de Madurez: {respuesta.porcentaje_madurez}%")
    print(f"Rendimiento Estimado: {respuesta.rendimiento_estimado_kg} kg de café cereza")
    if args.guardar_anotada:
        args.guardar_anotada.write_bytes(respuesta.imagen_anotada_bytes)
        print(f"Imagen anotada guardada en: {args.guardar_anotada}")
    if respuesta.advertencias:
        print("RESULTADO NO VALIDADO: no usar para decisiones de campo.")
    print("==========================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
