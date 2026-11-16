"""
Suite de contrato reutilizable para cualquier FruitDetectorPort.

Todo adaptador nuevo (YOLO, dos etapas, M0/M1...) debe pasarla:

    from contrato_detector import verificar_contrato
    verificar_contrato(mi_detector, imagen_con_frutos)
"""
from PIL import Image

from src.domain.models.coffee_cherry import DeteccionFruto
from src.domain.models.modelo_info import ModeloInfo


def _firma(dets):
    return [(d.estado, d.bbox, round(d.confianza, 6)) for d in dets]


def verificar_contrato(detector, imagen: Image.Image) -> None:
    # 1. Procedencia obligatoria y coherente.
    info = detector.info()
    assert isinstance(info, ModeloInfo)
    assert detector.es_simulado == info.es_simulado
    if info.es_simulado:
        assert not info.validado_con_datos_reales and info.advertencias

    # 2. Cajas dentro de la imagen, confianza en [0, 1], ids únicos.
    for umbral in (0.0, 0.4):
        dets = detector.detectar_frutos(imagen, umbral)
        assert all(isinstance(d, DeteccionFruto) for d in dets)
        for d in dets:
            assert 0 <= d.bbox.x_min < d.bbox.x_max <= imagen.width
            assert 0 <= d.bbox.y_min < d.bbox.y_max <= imagen.height
            assert 0.0 <= d.confianza <= 1.0
            # 3. Umbral respetado (tolerancia: los modelos trabajan en float32).
            assert d.confianza >= umbral - 1e-6
        assert len({d.id_deteccion for d in dets}) == len(dets)

    # 4. Más umbral nunca da más detecciones.
    assert len(detector.detectar_frutos(imagen, 0.9)) <= len(detector.detectar_frutos(imagen, 0.1))

    # 5. Determinista.
    assert _firma(detector.detectar_frutos(imagen, 0.4)) == _firma(detector.detectar_frutos(imagen, 0.4))

    # 6. Imágenes límite no revientan (uniforme, muy pequeña, escala de grises convertida).
    for extra in (Image.new("RGB", (64, 48), (128, 128, 128)), Image.new("RGB", (8, 8)), imagen.convert("L").convert("RGB")):
        detector.detectar_frutos(extra, 0.4)
