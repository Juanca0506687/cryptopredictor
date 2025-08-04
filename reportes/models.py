from django.db import models
from django.contrib.auth import get_user_model
from datetime import date, datetime
import json

User = get_user_model()

class TipoReporte(models.Model):
    nombre = models.CharField(max_length=100, unique=True)
    descripcion = models.TextField(blank=True, null=True)
    categoria = models.CharField(max_length=50, choices=[
        ('ventas', 'Ventas'),
        ('inventario', 'Inventario'),
        ('clientes', 'Clientes'),
        ('taller', 'Taller'),
        ('financiero', 'Financiero'),
    ])
    activo = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = 'Tipo de Reporte'
        verbose_name_plural = 'Tipos de Reportes'
        ordering = ['categoria', 'nombre']
    
    def __str__(self):
        return f"{self.nombre} ({self.get_categoria_display()})"

class Reporte(models.Model):
    ESTADOS = [
        ('generando', 'Generando'),
        ('completado', 'Completado'),
        ('error', 'Error'),
    ]
    
    FORMATOS = [
        ('pdf', 'PDF'),
        ('excel', 'Excel'),
        ('csv', 'CSV'),
    ]
    
    tipo_reporte = models.ForeignKey(TipoReporte, on_delete=models.CASCADE)
    usuario_solicita = models.ForeignKey(User, on_delete=models.CASCADE)
    fecha_solicitud = models.DateTimeField(auto_now_add=True)
    fecha_generacion = models.DateTimeField(blank=True, null=True)
    
    # Parámetros del reporte
    fecha_inicio = models.DateField(blank=True, null=True)
    fecha_fin = models.DateField(blank=True, null=True)
    parametros = models.JSONField(default=dict, blank=True)
    
    # Estado y archivo
    estado = models.CharField(max_length=20, choices=ESTADOS, default='generando')
    formato = models.CharField(max_length=10, choices=FORMATOS, default='pdf')
    archivo = models.FileField(upload_to='reportes/', blank=True, null=True)
    
    # Información adicional
    titulo = models.CharField(max_length=200)
    observaciones = models.TextField(blank=True, null=True)
    error_mensaje = models.TextField(blank=True, null=True)
    
    class Meta:
        verbose_name = 'Reporte'
        verbose_name_plural = 'Reportes'
        ordering = ['-fecha_solicitud']
    
    def __str__(self):
        return f"{self.titulo} - {self.usuario_solicita.username}"
    
    @property
    def esta_listo(self):
        return self.estado == 'completado' and self.archivo
    
    def marcar_completado(self, archivo_path):
        self.estado = 'completado'
        self.fecha_generacion = datetime.now()
        self.archivo = archivo_path
        self.save()
    
    def marcar_error(self, mensaje_error):
        self.estado = 'error'
        self.error_mensaje = mensaje_error
        self.save()

class ConfiguracionReporte(models.Model):
    """Configuraciones predefinidas para reportes frecuentes"""
    nombre = models.CharField(max_length=100)
    tipo_reporte = models.ForeignKey(TipoReporte, on_delete=models.CASCADE)
    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Configuración
    parametros_default = models.JSONField(default=dict)
    formato_default = models.CharField(max_length=10, choices=Reporte.FORMATOS, default='pdf')
    
    # Programación automática
    automatico = models.BooleanField(default=False)
    frecuencia_dias = models.IntegerField(blank=True, null=True)
    ultima_ejecucion = models.DateTimeField(blank=True, null=True)
    
    activo = models.BooleanField(default=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Configuración de Reporte'
        verbose_name_plural = 'Configuraciones de Reportes'
        ordering = ['nombre']
    
    def __str__(self):
        return f"{self.nombre} - {self.usuario.username}"

class EstadisticaVenta(models.Model):
    """Modelo para almacenar estadísticas precalculadas de ventas"""
    fecha = models.DateField()
    
    # Totales del día
    total_ventas = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    cantidad_ventas = models.IntegerField(default=0)
    total_cup = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_eur = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_mlc = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
    # Productos más vendidos
    monturas_vendidas = models.IntegerField(default=0)
    cristales_vendidos = models.IntegerField(default=0)
    
    # Clientes
    clientes_nuevos = models.IntegerField(default=0)
    clientes_recurrentes = models.IntegerField(default=0)
    
    # Vendedores
    vendedor_top = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True)
    ventas_vendedor_top = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Estadística de Venta'
        verbose_name_plural = 'Estadísticas de Ventas'
        unique_together = ['fecha']
        ordering = ['-fecha']
    
    def __str__(self):
        return f"Estadísticas {self.fecha} - {self.cantidad_ventas} ventas"

class EstadisticaInventario(models.Model):
    """Modelo para almacenar estadísticas de inventario"""
    fecha = models.DateField()
    
    # Monturas
    total_monturas = models.IntegerField(default=0)
    monturas_bajo_stock = models.IntegerField(default=0)
    valor_inventario_monturas = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
    # Cristales
    total_cristales = models.IntegerField(default=0)
    cristales_bajo_stock = models.IntegerField(default=0)
    valor_inventario_cristales = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
    # Movimientos
    entradas_dia = models.IntegerField(default=0)
    salidas_dia = models.IntegerField(default=0)
    
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Estadística de Inventario'
        verbose_name_plural = 'Estadísticas de Inventario'
        unique_together = ['fecha']
        ordering = ['-fecha']
    
    def __str__(self):
        return f"Inventario {self.fecha}"

class EstadisticaTaller(models.Model):
    """Modelo para almacenar estadísticas del taller"""
    fecha = models.DateField()
    
    # Órdenes de trabajo
    ordenes_creadas = models.IntegerField(default=0)
    ordenes_completadas = models.IntegerField(default=0)
    ordenes_pendientes = models.IntegerField(default=0)
    ordenes_atrasadas = models.IntegerField(default=0)
    
    # Tiempos
    tiempo_promedio_horas = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    eficiencia_porcentaje = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    
    # Calidad
    controles_aprobados = models.IntegerField(default=0)
    controles_rechazados = models.IntegerField(default=0)
    
    # Técnicos
    tecnico_mas_productivo = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True)
    ordenes_tecnico_top = models.IntegerField(default=0)
    
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Estadística de Taller'
        verbose_name_plural = 'Estadísticas de Taller'
        unique_together = ['fecha']
        ordering = ['-fecha']
    
    def __str__(self):
        return f"Taller {self.fecha} - {self.ordenes_completadas} completadas"
