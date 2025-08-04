from django.db import models
from django.core.validators import RegexValidator, MinValueValidator, MaxValueValidator
from django.contrib.auth import get_user_model
from datetime import date, timedelta

Usuario = get_user_model()


class Cliente(models.Model):
    """Modelo para clientes de la óptica"""
    
    cedula = models.CharField(
        max_length=11,
        unique=True,
        validators=[RegexValidator(r'^\d{11}$', 'La cédula debe tener 11 dígitos')],
        verbose_name='Cédula de Identidad'
    )
    nombre = models.CharField(
        max_length=100,
        verbose_name='Nombre'
    )
    apellidos = models.CharField(
        max_length=100,
        verbose_name='Apellidos'
    )
    fecha_nacimiento = models.DateField(
        verbose_name='Fecha de Nacimiento'
    )
    telefono = models.CharField(
        max_length=15,
        blank=True,
        null=True,
        verbose_name='Teléfono'
    )
    email = models.EmailField(
        blank=True,
        null=True,
        verbose_name='Email'
    )
    direccion = models.TextField(
        verbose_name='Dirección'
    )
    municipio = models.CharField(
        max_length=100,
        verbose_name='Municipio'
    )
    provincia = models.CharField(
        max_length=100,
        verbose_name='Provincia'
    )
    ocupacion = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name='Ocupación'
    )
    notas = models.TextField(
        blank=True,
        null=True,
        verbose_name='Notas'
    )
    activo = models.BooleanField(
        default=True,
        verbose_name='Activo'
    )
    fecha_registro = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Registro'
    )
    fecha_actualizacion = models.DateTimeField(
        auto_now=True,
        verbose_name='Fecha de Actualización'
    )
    registrado_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Registrado por'
    )
    
    class Meta:
        verbose_name = 'Cliente'
        verbose_name_plural = 'Clientes'
        ordering = ['apellidos', 'nombre']
    
    def __str__(self):
        return f"{self.nombre} {self.apellidos} ({self.cedula})"
    
    @property
    def nombre_completo(self):
        return f"{self.nombre} {self.apellidos}"
    
    @property
    def edad(self):
        today = date.today()
        return today.year - self.fecha_nacimiento.year - (
            (today.month, today.day) < (self.fecha_nacimiento.month, self.fecha_nacimiento.day)
        )
    
    @property
    def ultima_receta(self):
        """Obtener la receta más reciente"""
        return self.recetas.filter(activa=True).order_by('-fecha_examen').first()
    
    @property
    def necesita_revision(self):
        """Verificar si necesita revisión (más de 2 años desde la última receta)"""
        if not self.ultima_receta:
            return True
        return (date.today() - self.ultima_receta.fecha_examen) > timedelta(days=730)


class Receta(models.Model):
    """Recetas oftálmicas de los clientes"""
    
    cliente = models.ForeignKey(
        Cliente,
        on_delete=models.CASCADE,
        related_name='recetas'
    )
    fecha_examen = models.DateField(
        verbose_name='Fecha del Examen'
    )
    oftalmologo = models.CharField(
        max_length=200,
        verbose_name='Oftalmólogo'
    )
    centro_medico = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        verbose_name='Centro Médico'
    )
    
    # Ojo Derecho
    od_esfera = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(-20), MaxValueValidator(20)],
        verbose_name='OD Esfera'
    )
    od_cilindro = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(-10), MaxValueValidator(10)],
        verbose_name='OD Cilindro'
    )
    od_eje = models.IntegerField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0), MaxValueValidator(180)],
        verbose_name='OD Eje'
    )
    od_adicion = models.DecimalField(
        max_digits=3,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(0), MaxValueValidator(4)],
        verbose_name='OD Adición'
    )
    
    # Ojo Izquierdo
    oi_esfera = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(-20), MaxValueValidator(20)],
        verbose_name='OI Esfera'
    )
    oi_cilindro = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(-10), MaxValueValidator(10)],
        verbose_name='OI Cilindro'
    )
    oi_eje = models.IntegerField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0), MaxValueValidator(180)],
        verbose_name='OI Eje'
    )
    oi_adicion = models.DecimalField(
        max_digits=3,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(0), MaxValueValidator(4)],
        verbose_name='OI Adición'
    )
    
    # Distancia pupilar
    dp_lejana = models.DecimalField(
        max_digits=4,
        decimal_places=1,
        blank=True,
        null=True,
        validators=[MinValueValidator(50), MaxValueValidator(80)],
        verbose_name='DP Lejana (mm)'
    )
    dp_cercana = models.DecimalField(
        max_digits=4,
        decimal_places=1,
        blank=True,
        null=True,
        validators=[MinValueValidator(50), MaxValueValidator(80)],
        verbose_name='DP Cercana (mm)'
    )
    
    # Información adicional
    tipo_lente = models.CharField(
        max_length=20,
        choices=[
            ('monofocal', 'Monofocal'),
            ('bifocal', 'Bifocal'),
            ('progresivo', 'Progresivo'),
            ('ocupacional', 'Ocupacional'),
        ],
        default='monofocal',
        verbose_name='Tipo de Lente'
    )
    
    observaciones = models.TextField(
        blank=True,
        null=True,
        verbose_name='Observaciones'
    )
    
    activa = models.BooleanField(
        default=True,
        verbose_name='Activa'
    )
    
    fecha_creacion = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Creación'
    )
    
    creada_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Creada por'
    )
    
    class Meta:
        verbose_name = 'Receta'
        verbose_name_plural = 'Recetas'
        ordering = ['-fecha_examen']
    
    def __str__(self):
        return f"Receta de {self.cliente.nombre_completo} - {self.fecha_examen}"
    
    @property
    def es_progresiva(self):
        """Verificar si es una receta progresiva"""
        return self.tipo_lente == 'progresivo' or (
            self.od_adicion is not None or self.oi_adicion is not None
        )
    
    @property
    def graduacion_od_texto(self):
        """Texto de graduación del ojo derecho"""
        partes = []
        if self.od_esfera is not None:
            partes.append(f"Esf: {self.od_esfera:+.2f}")
        if self.od_cilindro is not None:
            partes.append(f"Cil: {self.od_cilindro:+.2f}")
        if self.od_eje is not None:
            partes.append(f"Eje: {self.od_eje}°")
        if self.od_adicion is not None:
            partes.append(f"Add: +{self.od_adicion:.2f}")
        return " ".join(partes) if partes else "Sin graduación"
    
    @property
    def graduacion_oi_texto(self):
        """Texto de graduación del ojo izquierdo"""
        partes = []
        if self.oi_esfera is not None:
            partes.append(f"Esf: {self.oi_esfera:+.2f}")
        if self.oi_cilindro is not None:
            partes.append(f"Cil: {self.oi_cilindro:+.2f}")
        if self.oi_eje is not None:
            partes.append(f"Eje: {self.oi_eje}°")
        if self.oi_adicion is not None:
            partes.append(f"Add: +{self.oi_adicion:.2f}")
        return " ".join(partes) if partes else "Sin graduación"


class HistorialCliente(models.Model):
    """Historial de cambios y eventos del cliente"""
    
    TIPO_CHOICES = [
        ('creacion', 'Creación'),
        ('actualizacion', 'Actualización'),
        ('nueva_receta', 'Nueva Receta'),
        ('venta', 'Venta'),
        ('reparacion', 'Reparación'),
        ('nota', 'Nota'),
    ]
    
    cliente = models.ForeignKey(
        Cliente,
        on_delete=models.CASCADE,
        related_name='historial'
    )
    tipo = models.CharField(
        max_length=20,
        choices=TIPO_CHOICES,
        verbose_name='Tipo'
    )
    descripcion = models.TextField(
        verbose_name='Descripción'
    )
    fecha = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha'
    )
    usuario = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Usuario'
    )
    
    class Meta:
        verbose_name = 'Historial de Cliente'
        verbose_name_plural = 'Historiales de Clientes'
        ordering = ['-fecha']
    
    def __str__(self):
        return f"{self.cliente.nombre_completo} - {self.get_tipo_display()} - {self.fecha}"


class NotificacionRevision(models.Model):
    """Notificaciones para revisiones periódicas"""
    
    cliente = models.ForeignKey(
        Cliente,
        on_delete=models.CASCADE,
        related_name='notificaciones'
    )
    mensaje = models.CharField(
        max_length=200,
        verbose_name='Mensaje'
    )
    fecha_programada = models.DateField(
        verbose_name='Fecha Programada'
    )
    enviada = models.BooleanField(
        default=False,
        verbose_name='Enviada'
    )
    fecha_envio = models.DateTimeField(
        blank=True,
        null=True,
        verbose_name='Fecha de Envío'
    )
    activa = models.BooleanField(
        default=True,
        verbose_name='Activa'
    )
    
    class Meta:
        verbose_name = 'Notificación de Revisión'
        verbose_name_plural = 'Notificaciones de Revisión'
        ordering = ['fecha_programada']
    
    def __str__(self):
        return f"Notificación para {self.cliente.nombre_completo} - {self.fecha_programada}"
