from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings

class Usuario(AbstractUser):
    ROLES = [
        ('administrador', 'Administrador'),
        ('vendedor', 'Vendedor'),
        ('tecnico', 'Técnico'),
    ]
    
    rol = models.CharField(max_length=20, choices=ROLES, default='vendedor')
    telefono = models.CharField(max_length=15, blank=True, null=True)
    direccion = models.TextField(blank=True, null=True)
    activo = models.BooleanField(default=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Usuario'
        verbose_name_plural = 'Usuarios'
    
    def __str__(self):
        return f"{self.username} ({self.get_rol_display()})"
    
    def puede_administrar(self):
        return self.rol == 'administrador'
    
    def puede_vender(self):
        return self.rol in ['administrador', 'vendedor']
    
    def puede_trabajar_taller(self):
        return self.rol in ['administrador', 'tecnico']
