"""
Excepciones de Dominio
"""
class DomainException(Exception):
    pass

class UmbralInvalidoException(DomainException):
    pass

class ImagenInvalidaException(DomainException):
    pass

class ParametrosInvalidosException(DomainException):
    pass

class ModeloNoDisponibleException(DomainException):
    pass
