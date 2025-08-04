from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings

class Usuario(AbstractUser):
    """Modelo de usuario personalizado con roles específicos para la óptica"""
    
    ROLES_CHOICES = [
        ('administrador', 'Administrador'),
        ('vendedor', 'Vendedor'),
        ('tecnico', 'Técnico'),
    ]
    
    rol = models.CharField(
        max_length=20,
        choices=ROLES_CHOICES,
        default='vendedor',
        verbose_name='Rol'
    )
    
    telefono = models.CharField(
        max_length=15,
        blank=True,
        null=True,
        verbose_name='Teléfono'
    )
    
    cedula = models.CharField(
        max_length=11,
        unique=True,
        blank=True,
        null=True,
        verbose_name='Cédula de Identidad'
    )
    
    direccion = models.TextField(
        blank=True,
        null=True,
        verbose_name='Dirección'
    )
    
    activo = models.BooleanField(
        default=True,
        verbose_name='Usuario Activo'
    )
    
    fecha_creacion = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Creación'
    )
    
    fecha_actualizacion = models.DateTimeField(
        auto_now=True,
        verbose_name='Fecha de Actualización'
    )
    
    class Meta:
        verbose_name = 'Usuario'
        verbose_name_plural = 'Usuarios'
        ordering = ['username']
    
    def __str__(self):
        return f"{self.username} ({self.get_rol_display()})"
    
    def es_administrador(self):
        return self.rol == 'administrador'
    
    def es_vendedor(self):
        return self.rol == 'vendedor'
    
    def es_tecnico(self):
        return self.rol == 'tecnico'
    
    def puede_crear_usuarios(self):
        return self.es_administrador()
    
    def puede_ver_reportes(self):
        return self.es_administrador() or self.es_vendedor()
    
    def puede_gestionar_taller(self):
        return self.es_administrador() or self.es_tecnico()
    
    def puede_realizar_ventas(self):
        return self.es_administrador() or self.es_vendedor()
    
    def puede_gestionar_productos(self):
        return self.es_administrador()


class PerfilUsuario(models.Model):
    """Perfil extendido del usuario con información adicional"""
    
    usuario = models.OneToOneField(
        Usuario,
        on_delete=models.CASCADE,
        related_name='perfil'
    )
    
    foto = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        verbose_name='Foto de Perfil'
    )
    
    biografia = models.TextField(
        blank=True,
        null=True,
        verbose_name='Biografía'
    )
    
    fecha_nacimiento = models.DateField(
        blank=True,
        null=True,
        verbose_name='Fecha de Nacimiento'
    )
    
    salario = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='Salario'
    )
    
    comision_ventas = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=0.00,
        verbose_name='Comisión por Ventas (%)'
    )
    
    class Meta:
        verbose_name = 'Perfil de Usuario'
        verbose_name_plural = 'Perfiles de Usuario'
    
    def __str__(self):
        return f"Perfil de {self.usuario.username}"


class SesionUsuario(models.Model):
    """Registro de sesiones de usuario para auditoría"""
    
    usuario = models.ForeignKey(
        Usuario,
        on_delete=models.CASCADE,
        related_name='sesiones'
    )
    
    fecha_inicio = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Inicio'
    )
    
    fecha_fin = models.DateTimeField(
        blank=True,
        null=True,
        verbose_name='Fecha de Fin'
    )
    
    ip_address = models.GenericIPAddressField(
        blank=True,
        null=True,
        verbose_name='Dirección IP'
    )
    
    user_agent = models.TextField(
        blank=True,
        null=True,
        verbose_name='User Agent'
    )
    
    activa = models.BooleanField(
        default=True,
        verbose_name='Sesión Activa'
    )
    
    class Meta:
        verbose_name = 'Sesión de Usuario'
        verbose_name_plural = 'Sesiones de Usuario'
        ordering = ['-fecha_inicio']
    
    def __str__(self):
        return f"Sesión de {self.usuario.username} - {self.fecha_inicio}"
