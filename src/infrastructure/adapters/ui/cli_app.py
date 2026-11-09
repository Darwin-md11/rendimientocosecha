"""
CLI Application Adapter

Ejemplos:
    python -m src.infrastructure.adapters.ui.cli_app                      # escena sintetica de demostracion
    python -m src.infrastructure.adapters.ui.cli_app --imagen foto.jpg    # foto real
    python -m src.infrastructure.adapters.ui.cli_app --detector mock      # detector simulado
"""
import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

from ....application.dtos.estimate_dto import SolicitudEstimacionDTO
from ....application.use_cases.estimate_yield_use_case import AnalizarImagenCosechaUseCase
from ....domain.exceptions.domain_exceptions import DomainException
from ...composition import construir_detector


def _imagen_demo_bytes() -> bytes:
    from ...ml.synthetic_data import generar_escena

    escena = generar_escena(np.random.default_rng(3))
    buf = io.BytesIO()
    Image.fromarray(escena.rgb).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run_cli(argv=None) -> int:
    p = argparse.ArgumentParser(description="SIKAFE: estimacion de cosecha de cafe.")
    p.add_argument("--imagen", type=Path, help="foto del cafeto (si se omite, usa una escena sintetica)")
    p.add_argument("--detector", choices=["ml", "mock"], default=None)
    p.add_argument("--peso-g", type=float, default=1.8, help="peso medio por cereza (g)")
    p.add_argument("--oclusion", type=float, default=1.25, help="factor de oclusion (>= 1.0)")
    p.add_argument("--plantas-muestra", type=int, default=1)
    p.add_argument("--plantas-lote", type=int, default=2500)
    p.add_argument("--umbral", type=float, default=0.40, help="umbral de confianza (0 a 1)")
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
        imagen_bytes = _imagen_demo_bytes()

    use_case = AnalizarImagenCosechaUseCase(detector_adapter=construir_detector(args.detector))
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

    print(f"Total cerezas detectadas: {respuesta.total_frutos}")
    print(f"Conteo por estado: {respuesta.conteo_por_estado}")
    print(f"Porcentaje de Madurez: {respuesta.porcentaje_madurez}%")
    print(f"Rendimiento Estimado: {respuesta.rendimiento_estimado_kg} kg de café cereza")
    if args.guardar_anotada:
        args.guardar_anotada.write_bytes(respuesta.imagen_anotada_bytes)
        print(f"Imagen anotada guardada en: {args.guardar_anotada}")
    print("==========================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
