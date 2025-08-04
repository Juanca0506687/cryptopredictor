from django.db import models
from django.core.validators import MinValueValidator
from datetime import date, timedelta
from django.utils import timezone

# Create your models here.

class OrdenTrabajo(models.Model):
    ESTADOS = [
        ('pendiente', 'Pendiente'),
        ('en_proceso', 'En Proceso'),
        ('pausada', 'Pausada'),
        ('control_calidad', 'Control de Calidad'),
        ('terminada', 'Terminada'),
        ('entregada', 'Entregada'),
        ('cancelada', 'Cancelada'),
    ]
    
    PRIORIDADES = [
        ('baja', 'Baja'),
        ('normal', 'Normal'),
        ('alta', 'Alta'),
        ('urgente', 'Urgente'),
    ]
    
    # Identificación
    numero_orden = models.CharField(max_length=20, unique=True, editable=False)
    venta = models.OneToOneField('ventas.Venta', on_delete=models.CASCADE, related_name='orden_trabajo')
    
    # Fechas
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_inicio = models.DateTimeField(blank=True, null=True)
    fecha_fin_estimada = models.DateTimeField(blank=True, null=True)
    fecha_fin_real = models.DateTimeField(blank=True, null=True)
    
    # Estado y prioridad
    estado = models.CharField(max_length=20, choices=ESTADOS, default='pendiente')
    prioridad = models.CharField(max_length=10, choices=PRIORIDADES, default='normal')
    
    # Asignación
    tecnico_asignado = models.ForeignKey('usuarios.Usuario', on_delete=models.SET_NULL, blank=True, null=True, related_name='ordenes_asignadas')
    supervisor = models.ForeignKey('usuarios.Usuario', on_delete=models.SET_NULL, blank=True, null=True, related_name='ordenes_supervisadas')
    
    # Información técnica
    observaciones_tecnicas = models.TextField(blank=True, null=True)
    observaciones_calidad = models.TextField(blank=True, null=True)
    tiempo_estimado_horas = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    tiempo_real_horas = models.DecimalField(max_digits=5, decimal_places=2, default=0, blank=True, null=True)
    
    # Control
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
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
                fecha_creacion__year=self.fecha_creacion.year if self.fecha_creacion else timezone.now().year
            ).count() + 1
            year = self.fecha_creacion.year if self.fecha_creacion else timezone.now().year
            self.numero_orden = f"OT{year}{ultimo_numero:04d}"
        super().save(*args, **kwargs)
    
    @property
    def dias_transcurridos(self):
        if self.fecha_inicio:
            fin = self.fecha_fin_real or timezone.now()
            return (fin.date() - self.fecha_inicio.date()).days
        return 0
    
    @property
    def esta_atrasada(self):
        if self.fecha_fin_estimada and self.estado not in ['terminada', 'entregada', 'cancelada']:
            return timezone.now() > self.fecha_fin_estimada
        return False
    
    @property
    def porcentaje_completado(self):
        tareas_total = self.tareas.count()
        if tareas_total == 0:
            return 0
        tareas_completadas = self.tareas.filter(completada=True).count()
        return int((tareas_completadas / tareas_total) * 100)

class TareaTaller(models.Model):
    TIPOS_TAREA = [
        ('medicion', 'Medición'),
        ('corte', 'Corte de Cristal'),
        ('biselado', 'Biselado'),
        ('montaje', 'Montaje'),
        ('ajuste', 'Ajuste'),
        ('control_calidad', 'Control de Calidad'),
        ('limpieza', 'Limpieza Final'),
        ('empaque', 'Empaque'),
    ]
    
    orden_trabajo = models.ForeignKey(OrdenTrabajo, on_delete=models.CASCADE, related_name='tareas')
    tipo_tarea = models.CharField(max_length=20, choices=TIPOS_TAREA)
    descripcion = models.CharField(max_length=200)
    
    # Secuencia
    orden_secuencia = models.IntegerField(default=1)
    
    # Asignación y tiempos
    tecnico_asignado = models.ForeignKey('usuarios.Usuario', on_delete=models.SET_NULL, blank=True, null=True)
    tiempo_estimado_minutos = models.IntegerField(default=30)
    tiempo_real_minutos = models.IntegerField(blank=True, null=True)
    
    # Estado
    completada = models.BooleanField(default=False)
    fecha_inicio = models.DateTimeField(blank=True, null=True)
    fecha_fin = models.DateTimeField(blank=True, null=True)
    
    # Observaciones
    observaciones = models.TextField(blank=True, null=True)
    requiere_supervision = models.BooleanField(default=False)
    aprobada_por_supervisor = models.BooleanField(default=False)
    
    # Control
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Tarea de Taller'
        verbose_name_plural = 'Tareas de Taller'
        ordering = ['orden_trabajo', 'orden_secuencia']
        unique_together = ['orden_trabajo', 'orden_secuencia']
    
    def __str__(self):
        return f"{self.orden_trabajo.numero_orden} - {self.get_tipo_tarea_display()}"
    
    def puede_iniciar(self):
        """Verifica si la tarea puede iniciarse (tareas anteriores completadas)"""
        tareas_anteriores = TareaTaller.objects.filter(
            orden_trabajo=self.orden_trabajo,
            orden_secuencia__lt=self.orden_secuencia
        )
        return all(tarea.completada for tarea in tareas_anteriores)
    
    def marcar_completada(self, tecnico, observaciones=None):
        """Marca la tarea como completada"""
        from django.utils import timezone
        self.completada = True
        self.fecha_fin = timezone.now()
        if observaciones:
            self.observaciones = observaciones
        if not self.fecha_inicio:
            self.fecha_inicio = timezone.now()
        if not self.tiempo_real_minutos and self.fecha_inicio:
            diferencia = self.fecha_fin - self.fecha_inicio
            self.tiempo_real_minutos = int(diferencia.total_seconds() / 60)
        self.save()

class ControlCalidad(models.Model):
    RESULTADOS = [
        ('aprobado', 'Aprobado'),
        ('rechazado', 'Rechazado'),
        ('rehacer', 'Rehacer'),
    ]
    
    orden_trabajo = models.ForeignKey(OrdenTrabajo, on_delete=models.CASCADE, related_name='controles_calidad')
    inspector = models.ForeignKey('usuarios.Usuario', on_delete=models.CASCADE)
    fecha_inspeccion = models.DateTimeField(auto_now_add=True)
    
    # Criterios de evaluación
    graduacion_correcta = models.BooleanField(default=False)
    centrado_correcto = models.BooleanField(default=False)
    acabado_satisfactorio = models.BooleanField(default=False)
    limpieza_adecuada = models.BooleanField(default=False)
    ajuste_correcto = models.BooleanField(default=False)
    
    # Resultado
    resultado = models.CharField(max_length=20, choices=RESULTADOS)
    observaciones = models.TextField(blank=True, null=True)
    defectos_encontrados = models.TextField(blank=True, null=True)
    
    # Mediciones técnicas
    dp_od = models.DecimalField(max_digits=4, decimal_places=1, blank=True, null=True, verbose_name="DP OD")
    dp_oi = models.DecimalField(max_digits=4, decimal_places=1, blank=True, null=True, verbose_name="DP OI")
    altura_od = models.DecimalField(max_digits=4, decimal_places=1, blank=True, null=True, verbose_name="Altura OD")
    altura_oi = models.DecimalField(max_digits=4, decimal_places=1, blank=True, null=True, verbose_name="Altura OI")
    
    class Meta:
        verbose_name = 'Control de Calidad'
        verbose_name_plural = 'Controles de Calidad'
        ordering = ['-fecha_inspeccion']
    
    def __str__(self):
        return f"Control {self.orden_trabajo.numero_orden} - {self.get_resultado_display()}"
    
    @property
    def aprobado(self):
        return self.resultado == 'aprobado'
    
    @property
    def puntuacion_calidad(self):
        """Calcula puntuación de calidad basada en criterios"""
        criterios = [
            self.graduacion_correcta,
            self.centrado_correcto,
            self.acabado_satisfactorio,
            self.limpieza_adecuada,
            self.ajuste_correcto,
        ]
        return (sum(criterios) / len(criterios)) * 100

class RecursoTaller(models.Model):
    TIPOS_RECURSO = [
        ('maquina', 'Máquina'),
        ('herramienta', 'Herramienta'),
        ('equipo', 'Equipo'),
        ('material', 'Material'),
    ]
    
    ESTADOS = [
        ('disponible', 'Disponible'),
        ('en_uso', 'En Uso'),
        ('mantenimiento', 'En Mantenimiento'),
        ('averiado', 'Averiado'),
        ('fuera_servicio', 'Fuera de Servicio'),
    ]
    
    nombre = models.CharField(max_length=100)
    tipo = models.CharField(max_length=20, choices=TIPOS_RECURSO)
    codigo = models.CharField(max_length=50, unique=True)
    descripcion = models.TextField(blank=True, null=True)
    
    # Estado y ubicación
    estado = models.CharField(max_length=20, choices=ESTADOS, default='disponible')
    ubicacion = models.CharField(max_length=100, blank=True, null=True)
    
    # Información técnica
    marca = models.CharField(max_length=100, blank=True, null=True)
    modelo = models.CharField(max_length=100, blank=True, null=True)
    numero_serie = models.CharField(max_length=100, blank=True, null=True)
    
    # Mantenimiento
    fecha_ultimo_mantenimiento = models.DateField(blank=True, null=True)
    frecuencia_mantenimiento_dias = models.IntegerField(default=90)
    
    # Control
    activo = models.BooleanField(default=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Recurso de Taller'
        verbose_name_plural = 'Recursos de Taller'
        ordering = ['nombre']
    
    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    
    @property
    def necesita_mantenimiento(self):
        if self.fecha_ultimo_mantenimiento:
            proxima_fecha = self.fecha_ultimo_mantenimiento + timedelta(days=self.frecuencia_mantenimiento_dias)
            return date.today() >= proxima_fecha
        return True  # Si nunca ha tenido mantenimiento
    
    def esta_disponible(self):
        return self.estado == 'disponible' and self.activo

class UsoRecurso(models.Model):
    recurso = models.ForeignKey(RecursoTaller, on_delete=models.CASCADE, related_name='usos')
    orden_trabajo = models.ForeignKey(OrdenTrabajo, on_delete=models.CASCADE, related_name='recursos_utilizados')
    tarea = models.ForeignKey(TareaTaller, on_delete=models.CASCADE, blank=True, null=True)
    
    usuario = models.ForeignKey('usuarios.Usuario', on_delete=models.CASCADE)
    fecha_inicio = models.DateTimeField(auto_now_add=True)
    fecha_fin = models.DateTimeField(blank=True, null=True)
    
    observaciones = models.TextField(blank=True, null=True)
    
    class Meta:
        verbose_name = 'Uso de Recurso'
        verbose_name_plural = 'Usos de Recursos'
        ordering = ['-fecha_inicio']
    
    def __str__(self):
        return f"{self.recurso.nombre} - {self.orden_trabajo.numero_orden}"
    
    @property
    def tiempo_uso_minutos(self):
        if self.fecha_fin:
            diferencia = self.fecha_fin - self.fecha_inicio
            return int(diferencia.total_seconds() / 60)
        return None

class MantenimientoRecurso(models.Model):
    TIPOS_MANTENIMIENTO = [
        ('preventivo', 'Preventivo'),
        ('correctivo', 'Correctivo'),
        ('calibracion', 'Calibración'),
        ('limpieza', 'Limpieza'),
    ]
    
    recurso = models.ForeignKey(RecursoTaller, on_delete=models.CASCADE, related_name='mantenimientos')
    tipo = models.CharField(max_length=20, choices=TIPOS_MANTENIMIENTO)
    fecha_mantenimiento = models.DateTimeField(auto_now_add=True)
    tecnico = models.ForeignKey('usuarios.Usuario', on_delete=models.CASCADE)
    
    descripcion = models.TextField()
    costo = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    tiempo_fuera_servicio_horas = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    
    # Repuestos utilizados
    repuestos_utilizados = models.TextField(blank=True, null=True)
    
    # Resultado
    completado = models.BooleanField(default=True)
    observaciones = models.TextField(blank=True, null=True)
    
    class Meta:
        verbose_name = 'Mantenimiento de Recurso'
        verbose_name_plural = 'Mantenimientos de Recursos'
        ordering = ['-fecha_mantenimiento']
    
    def __str__(self):
        return f"{self.get_tipo_display()} - {self.recurso.nombre} - {self.fecha_mantenimiento.date()}"
