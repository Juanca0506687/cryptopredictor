from django.db import models
from django.core.validators import MinValueValidator
from django.contrib.auth import get_user_model
from django.utils import timezone
from ventas.models import Venta
from productos.models import Producto
from datetime import date, timedelta

Usuario = get_user_model()


class OrdenTrabajo(models.Model):
    """Órdenes de trabajo para el taller"""
    
    ESTADO_CHOICES = [
        ('pendiente', 'Pendiente'),
        ('en_proceso', 'En Proceso'),
        ('pausada', 'Pausada'),
        ('completada', 'Completada'),
        ('cancelada', 'Cancelada'),
    ]
    
    PRIORIDAD_CHOICES = [
        ('baja', 'Baja'),
        ('normal', 'Normal'),
        ('alta', 'Alta'),
        ('urgente', 'Urgente'),
    ]
    
    numero_orden = models.CharField(
        max_length=20,
        unique=True,
        verbose_name='Número de Orden'
    )
    venta = models.ForeignKey(
        Venta,
        on_delete=models.CASCADE,
        related_name='ordenes_trabajo'
    )
    fecha_creacion = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Creación'
    )
    fecha_inicio = models.DateTimeField(
        blank=True,
        null=True,
        verbose_name='Fecha de Inicio'
    )
    fecha_fin_estimada = models.DateField(
        verbose_name='Fecha de Fin Estimada'
    )
    fecha_fin_real = models.DateTimeField(
        blank=True,
        null=True,
        verbose_name='Fecha de Fin Real'
    )
    estado = models.CharField(
        max_length=20,
        choices=ESTADO_CHOICES,
        default='pendiente',
        verbose_name='Estado'
    )
    prioridad = models.CharField(
        max_length=20,
        choices=PRIORIDAD_CHOICES,
        default='normal',
        verbose_name='Prioridad'
    )
    tecnico_asignado = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        blank=True,
        null=True,
        limit_choices_to={'rol': 'tecnico'},
        related_name='ordenes_asignadas',
        verbose_name='Técnico Asignado'
    )
    descripcion_trabajo = models.TextField(
        verbose_name='Descripción del Trabajo'
    )
    observaciones = models.TextField(
        blank=True,
        null=True,
        verbose_name='Observaciones'
    )
    creada_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        related_name='ordenes_creadas',
        verbose_name='Creada por'
    )
    
    class Meta:
        verbose_name = 'Orden de Trabajo'
        verbose_name_plural = 'Órdenes de Trabajo'
        ordering = ['-fecha_creacion']
    
    def __str__(self):
        return f"Orden {self.numero_orden} - {self.venta.cliente.nombre_completo}"
    
    def save(self, *args, **kwargs):
        if not self.numero_orden:
            # Generar número de orden automático
            ultimo_numero = OrdenTrabajo.objects.filter(
                numero_orden__startswith=f'OT{self.fecha_creacion.year}'
            ).count() + 1
            self.numero_orden = f'OT{self.fecha_creacion.year}{ultimo_numero:06d}'
        
        super().save(*args, **kwargs)
    
    @property
    def dias_transcurridos(self):
        """Días transcurridos desde la creación"""
        return (date.today() - self.fecha_creacion.date()).days
    
    @property
    def esta_retrasada(self):
        """Verificar si la orden está retrasada"""
        return date.today() > self.fecha_fin_estimada and self.estado != 'completada'
    
    @property
    def tiempo_estimado_restante(self):
        """Tiempo estimado restante en días"""
        if self.estado == 'completada':
            return 0
        return (self.fecha_fin_estimada - date.today()).days
    
    def puede_iniciar(self):
        """Verificar si la orden puede iniciar"""
        return self.estado == 'pendiente' and self.tecnico_asignado is not None
    
    def iniciar_trabajo(self, usuario):
        """Iniciar el trabajo en la orden"""
        if self.puede_iniciar():
            self.estado = 'en_proceso'
            self.fecha_inicio = timezone.now()
            self.save()
            
            # Registrar en historial
            HistorialOrden.objects.create(
                orden=self,
                accion='inicio',
                descripcion='Trabajo iniciado',
                usuario=usuario
            )
    
    def completar_trabajo(self, usuario, observaciones=''):
        """Completar el trabajo en la orden"""
        if self.estado == 'en_proceso':
            self.estado = 'completada'
            self.fecha_fin_real = timezone.now()
            if observaciones:
                self.observaciones = observaciones
            self.save()
            
            # Actualizar estado de la venta
            self.venta.estado = 'lista'
            self.venta.save()
            
            # Registrar en historial
            HistorialOrden.objects.create(
                orden=self,
                accion='completado',
                descripcion='Trabajo completado',
                usuario=usuario
            )


class TipoTrabajo(models.Model):
    """Tipos de trabajo que se realizan en el taller"""
    
    nombre = models.CharField(
        max_length=100,
        unique=True,
        verbose_name='Nombre'
    )
    descripcion = models.TextField(
        blank=True,
        null=True,
        verbose_name='Descripción'
    )
    tiempo_estimado_horas = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Tiempo Estimado (horas)'
    )
    requiere_graduacion = models.BooleanField(
        default=False,
        verbose_name='Requiere Graduación'
    )
    activo = models.BooleanField(
        default=True,
        verbose_name='Activo'
    )
    
    class Meta:
        verbose_name = 'Tipo de Trabajo'
        verbose_name_plural = 'Tipos de Trabajo'
        ordering = ['nombre']
    
    def __str__(self):
        return self.nombre


class DetalleOrden(models.Model):
    """Detalles específicos de cada orden de trabajo"""
    
    orden = models.ForeignKey(
        OrdenTrabajo,
        on_delete=models.CASCADE,
        related_name='detalles'
    )
    tipo_trabajo = models.ForeignKey(
        TipoTrabajo,
        on_delete=models.PROTECT
    )
    producto = models.ForeignKey(
        Producto,
        on_delete=models.PROTECT,
        blank=True,
        null=True,
        verbose_name='Producto'
    )
    
    # Graduación específica para este trabajo
    od_esfera = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='OD Esfera'
    )
    od_cilindro = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='OD Cilindro'
    )
    od_eje = models.IntegerField(
        blank=True,
        null=True,
        verbose_name='OD Eje'
    )
    od_adicion = models.DecimalField(
        max_digits=3,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='OD Adición'
    )
    
    oi_esfera = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='OI Esfera'
    )
    oi_cilindro = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='OI Cilindro'
    )
    oi_eje = models.IntegerField(
        blank=True,
        null=True,
        verbose_name='OI Eje'
    )
    oi_adicion = models.DecimalField(
        max_digits=3,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='OI Adición'
    )
    
    dp_lejana = models.DecimalField(
        max_digits=4,
        decimal_places=1,
        blank=True,
        null=True,
        verbose_name='DP Lejana (mm)'
    )
    dp_cercana = models.DecimalField(
        max_digits=4,
        decimal_places=1,
        blank=True,
        null=True,
        verbose_name='DP Cercana (mm)'
    )
    
    especificaciones = models.TextField(
        blank=True,
        null=True,
        verbose_name='Especificaciones'
    )
    
    completado = models.BooleanField(
        default=False,
        verbose_name='Completado'
    )
    
    tiempo_real_horas = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(0)],
        verbose_name='Tiempo Real (horas)'
    )
    
    class Meta:
        verbose_name = 'Detalle de Orden'
        verbose_name_plural = 'Detalles de Orden'
    
    def __str__(self):
        return f"{self.orden.numero_orden} - {self.tipo_trabajo.nombre}"


class HistorialOrden(models.Model):
    """Historial de cambios en las órdenes de trabajo"""
    
    ACCION_CHOICES = [
        ('creacion', 'Creación'),
        ('asignacion', 'Asignación'),
        ('inicio', 'Inicio'),
        ('pausa', 'Pausa'),
        ('reanudacion', 'Reanudación'),
        ('completado', 'Completado'),
        ('cancelacion', 'Cancelación'),
        ('modificacion', 'Modificación'),
    ]
    
    orden = models.ForeignKey(
        OrdenTrabajo,
        on_delete=models.CASCADE,
        related_name='historial'
    )
    fecha = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha'
    )
    accion = models.CharField(
        max_length=20,
        choices=ACCION_CHOICES,
        verbose_name='Acción'
    )
    descripcion = models.TextField(
        verbose_name='Descripción'
    )
    usuario = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Usuario'
    )
    
    class Meta:
        verbose_name = 'Historial de Orden'
        verbose_name_plural = 'Historiales de Órdenes'
        ordering = ['-fecha']
    
    def __str__(self):
        return f"{self.orden.numero_orden} - {self.get_accion_display()}"


class ControlCalidad(models.Model):
    """Control de calidad de los trabajos realizados"""
    
    RESULTADO_CHOICES = [
        ('aprobado', 'Aprobado'),
        ('rechazado', 'Rechazado'),
        ('requiere_ajuste', 'Requiere Ajuste'),
    ]
    
    orden = models.OneToOneField(
        OrdenTrabajo,
        on_delete=models.CASCADE,
        related_name='control_calidad'
    )
    fecha_revision = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Revisión'
    )
    resultado = models.CharField(
        max_length=20,
        choices=RESULTADO_CHOICES,
        verbose_name='Resultado'
    )
    observaciones = models.TextField(
        verbose_name='Observaciones'
    )
    defectos_encontrados = models.TextField(
        blank=True,
        null=True,
        verbose_name='Defectos Encontrados'
    )
    acciones_correctivas = models.TextField(
        blank=True,
        null=True,
        verbose_name='Acciones Correctivas'
    )
    revisado_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Revisado por'
    )
    
    class Meta:
        verbose_name = 'Control de Calidad'
        verbose_name_plural = 'Controles de Calidad'
        ordering = ['-fecha_revision']
    
    def __str__(self):
        return f"Control de Calidad - {self.orden.numero_orden}"


class Herramienta(models.Model):
    """Herramientas y equipos del taller"""
    
    ESTADO_CHOICES = [
        ('disponible', 'Disponible'),
        ('en_uso', 'En Uso'),
        ('mantenimiento', 'En Mantenimiento'),
        ('fuera_servicio', 'Fuera de Servicio'),
    ]
    
    nombre = models.CharField(
        max_length=100,
        verbose_name='Nombre'
    )
    codigo = models.CharField(
        max_length=20,
        unique=True,
        verbose_name='Código'
    )
    descripcion = models.TextField(
        blank=True,
        null=True,
        verbose_name='Descripción'
    )
    marca = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        verbose_name='Marca'
    )
    modelo = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        verbose_name='Modelo'
    )
    numero_serie = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        verbose_name='Número de Serie'
    )
    fecha_adquisicion = models.DateField(
        blank=True,
        null=True,
        verbose_name='Fecha de Adquisición'
    )
    estado = models.CharField(
        max_length=20,
        choices=ESTADO_CHOICES,
        default='disponible',
        verbose_name='Estado'
    )
    ubicacion = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name='Ubicación'
    )
    
    class Meta:
        verbose_name = 'Herramienta'
        verbose_name_plural = 'Herramientas'
        ordering = ['nombre']
    
    def __str__(self):
        return f"{self.codigo} - {self.nombre}"


class UsoHerramienta(models.Model):
    """Registro de uso de herramientas"""
    
    herramienta = models.ForeignKey(
        Herramienta,
        on_delete=models.CASCADE,
        related_name='usos'
    )
    orden = models.ForeignKey(
        OrdenTrabajo,
        on_delete=models.CASCADE,
        related_name='herramientas_usadas'
    )
    usuario = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Usuario'
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
    observaciones = models.TextField(
        blank=True,
        null=True,
        verbose_name='Observaciones'
    )
    
    class Meta:
        verbose_name = 'Uso de Herramienta'
        verbose_name_plural = 'Usos de Herramientas'
        ordering = ['-fecha_inicio']
    
    def __str__(self):
        return f"{self.herramienta.nombre} - {self.orden.numero_orden}"


class MantenimientoHerramienta(models.Model):
    """Mantenimientos realizados a las herramientas"""
    
    TIPO_CHOICES = [
        ('preventivo', 'Preventivo'),
        ('correctivo', 'Correctivo'),
        ('calibracion', 'Calibración'),
    ]
    
    herramienta = models.ForeignKey(
        Herramienta,
        on_delete=models.CASCADE,
        related_name='mantenimientos'
    )
    fecha = models.DateField(
        verbose_name='Fecha'
    )
    tipo = models.CharField(
        max_length=20,
        choices=TIPO_CHOICES,
        verbose_name='Tipo'
    )
    descripcion = models.TextField(
        verbose_name='Descripción'
    )
    costo = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        blank=True,
        null=True,
        validators=[MinValueValidator(0)],
        verbose_name='Costo'
    )
    proveedor = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name='Proveedor'
    )
    realizado_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Realizado por'
    )
    
    class Meta:
        verbose_name = 'Mantenimiento de Herramienta'
        verbose_name_plural = 'Mantenimientos de Herramientas'
        ordering = ['-fecha']
    
    def __str__(self):
        return f"Mantenimiento {self.get_tipo_display()} - {self.herramienta.nombre}"
