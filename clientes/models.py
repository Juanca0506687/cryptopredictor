from django.db import models
from django.core.validators import RegexValidator
from datetime import date, timedelta

class Cliente(models.Model):
    TIPOS_DOCUMENTO = [
        ('ci', 'Carnet de Identidad'),
        ('pasaporte', 'Pasaporte'),
        ('otro', 'Otro'),
    ]
    
    # Datos personales
    numero_documento = models.CharField(max_length=20, unique=True)
    tipo_documento = models.CharField(max_length=10, choices=TIPOS_DOCUMENTO, default='ci')
    nombres = models.CharField(max_length=100)
    apellidos = models.CharField(max_length=100)
    fecha_nacimiento = models.DateField(blank=True, null=True)
    
    # Contacto
    telefono = models.CharField(max_length=15, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    direccion = models.TextField(blank=True, null=True)
    
    # Información médica
    alergias = models.TextField(blank=True, null=True, help_text="Alergias conocidas")
    observaciones_medicas = models.TextField(blank=True, null=True)
    
    # Control
    activo = models.BooleanField(default=True)
    fecha_registro = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Cliente'
        verbose_name_plural = 'Clientes'
        ordering = ['apellidos', 'nombres']
    
    def __str__(self):
        return f"{self.apellidos}, {self.nombres} ({self.numero_documento})"
    
    @property
    def nombre_completo(self):
        return f"{self.nombres} {self.apellidos}"
    
    @property
    def edad(self):
        if self.fecha_nacimiento:
            today = date.today()
            return today.year - self.fecha_nacimiento.year - ((today.month, today.day) < (self.fecha_nacimiento.month, self.fecha_nacimiento.day))
        return None
    
    def ultima_compra(self):
        from ventas.models import Venta
        return Venta.objects.filter(cliente=self).order_by('-fecha').first()
    
    def total_compras(self):
        from ventas.models import Venta
        return Venta.objects.filter(cliente=self).count()

class RecetaMedica(models.Model):
    cliente = models.ForeignKey(Cliente, on_delete=models.CASCADE, related_name='recetas')
    fecha_receta = models.DateField()
    medico = models.CharField(max_length=200, blank=True, null=True)
    centro_medico = models.CharField(max_length=200, blank=True, null=True)
    
    # Ojo derecho
    od_esfera = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, verbose_name="OD Esfera")
    od_cilindro = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, verbose_name="OD Cilindro")
    od_eje = models.IntegerField(default=0, verbose_name="OD Eje")
    od_adicion = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, verbose_name="OD Adición", blank=True, null=True)
    
    # Ojo izquierdo
    oi_esfera = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, verbose_name="OI Esfera")
    oi_cilindro = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, verbose_name="OI Cilindro")
    oi_eje = models.IntegerField(default=0, verbose_name="OI Eje")
    oi_adicion = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, verbose_name="OI Adición", blank=True, null=True)
    
    # Distancia pupilar
    dp_lejos = models.DecimalField(max_digits=4, decimal_places=1, blank=True, null=True, verbose_name="DP Lejos")
    dp_cerca = models.DecimalField(max_digits=4, decimal_places=1, blank=True, null=True, verbose_name="DP Cerca")
    
    # Información adicional
    tipo_lente = models.CharField(max_length=50, blank=True, null=True)
    observaciones = models.TextField(blank=True, null=True)
    activa = models.BooleanField(default=True)
    
    # Control
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Receta Médica'
        verbose_name_plural = 'Recetas Médicas'
        ordering = ['-fecha_receta']
    
    def __str__(self):
        return f"Receta {self.cliente.nombre_completo} - {self.fecha_receta}"
    
    @property
    def necesita_revision(self):
        """Verifica si la receta necesita revisión (más de 2 años)"""
        if self.fecha_receta:
            return date.today() > (self.fecha_receta + timedelta(days=730))
        return False
    
    @property
    def graduacion_od(self):
        return f"Esf: {self.od_esfera:+.2f} Cil: {self.od_cilindro:+.2f} Eje: {self.od_eje}°"
    
    @property
    def graduacion_oi(self):
        return f"Esf: {self.oi_esfera:+.2f} Cil: {self.oi_cilindro:+.2f} Eje: {self.oi_eje}°"

class Notificacion(models.Model):
    TIPOS_NOTIFICACION = [
        ('revision_receta', 'Revisión de Receta'),
        ('entrega_pedido', 'Entrega de Pedido'),
        ('promocion', 'Promoción'),
        ('recordatorio', 'Recordatorio'),
    ]
    
    cliente = models.ForeignKey(Cliente, on_delete=models.CASCADE, related_name='notificaciones')
    tipo = models.CharField(max_length=20, choices=TIPOS_NOTIFICACION)
    titulo = models.CharField(max_length=200)
    mensaje = models.TextField()
    fecha_programada = models.DateTimeField()
    enviada = models.BooleanField(default=False)
    fecha_envio = models.DateTimeField(blank=True, null=True)
    
    # Control
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Notificación'
        verbose_name_plural = 'Notificaciones'
        ordering = ['-fecha_programada']
    
    def __str__(self):
        return f"{self.cliente.nombre_completo} - {self.titulo}"
