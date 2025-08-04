from django.db import models
from django.core.validators import MinValueValidator
from django.contrib.auth import get_user_model
from productos.models import Producto
from clientes.models import Cliente, Receta
import uuid

Usuario = get_user_model()


class TasaCambio(models.Model):
    """Tasas de cambio para diferentes monedas"""
    
    MONEDA_CHOICES = [
        ('USD', 'Dólar Estadounidense'),
        ('EUR', 'Euro'),
        ('MLC', 'Moneda Libremente Convertible'),
    ]
    
    moneda = models.CharField(
        max_length=3,
        choices=MONEDA_CHOICES,
        unique=True,
        verbose_name='Moneda'
    )
    tasa_cup = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Tasa a CUP'
    )
    fecha_actualizacion = models.DateTimeField(
        auto_now=True,
        verbose_name='Fecha de Actualización'
    )
    actualizada_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Actualizada por'
    )
    
    class Meta:
        verbose_name = 'Tasa de Cambio'
        verbose_name_plural = 'Tasas de Cambio'
    
    def __str__(self):
        return f"1 {self.moneda} = {self.tasa_cup} CUP"


class Venta(models.Model):
    """Ventas realizadas en la óptica"""
    
    ESTADO_CHOICES = [
        ('borrador', 'Borrador'),
        ('confirmada', 'Confirmada'),
        ('en_taller', 'En Taller'),
        ('lista', 'Lista para Entrega'),
        ('entregada', 'Entregada'),
        ('cancelada', 'Cancelada'),
    ]
    
    numero_factura = models.CharField(
        max_length=20,
        unique=True,
        verbose_name='Número de Factura'
    )
    cliente = models.ForeignKey(
        Cliente,
        on_delete=models.PROTECT,
        related_name='ventas'
    )
    receta = models.ForeignKey(
        Receta,
        on_delete=models.PROTECT,
        blank=True,
        null=True,
        verbose_name='Receta'
    )
    fecha_venta = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Venta'
    )
    fecha_entrega_estimada = models.DateField(
        blank=True,
        null=True,
        verbose_name='Fecha de Entrega Estimada'
    )
    fecha_entrega_real = models.DateField(
        blank=True,
        null=True,
        verbose_name='Fecha de Entrega Real'
    )
    estado = models.CharField(
        max_length=20,
        choices=ESTADO_CHOICES,
        default='borrador',
        verbose_name='Estado'
    )
    moneda_venta = models.CharField(
        max_length=3,
        choices=[
            ('CUP', 'Peso Cubano'),
            ('USD', 'Dólar Estadounidense'),
            ('EUR', 'Euro'),
            ('MLC', 'Moneda Libremente Convertible'),
        ],
        default='CUP',
        verbose_name='Moneda de Venta'
    )
    subtotal = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        verbose_name='Subtotal'
    )
    descuento_porcentaje = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        verbose_name='Descuento (%)'
    )
    descuento_monto = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        verbose_name='Descuento (Monto)'
    )
    total = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        verbose_name='Total'
    )
    observaciones = models.TextField(
        blank=True,
        null=True,
        verbose_name='Observaciones'
    )
    vendedor = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        related_name='ventas_realizadas',
        verbose_name='Vendedor'
    )
    
    class Meta:
        verbose_name = 'Venta'
        verbose_name_plural = 'Ventas'
        ordering = ['-fecha_venta']
    
    def __str__(self):
        return f"Factura {self.numero_factura} - {self.cliente.nombre_completo}"
    
    def save(self, *args, **kwargs):
        if not self.numero_factura:
            # Generar número de factura automático
            ultimo_numero = Venta.objects.filter(
                numero_factura__startswith=f'F{self.fecha_venta.year}'
            ).count() + 1
            self.numero_factura = f'F{self.fecha_venta.year}{ultimo_numero:06d}'
        
        # Calcular totales
        self.calcular_totales()
        super().save(*args, **kwargs)
    
    def calcular_totales(self):
        """Calcular subtotal, descuentos y total"""
        self.subtotal = sum(item.subtotal for item in self.items.all())
        
        if self.descuento_porcentaje > 0:
            self.descuento_monto = (self.subtotal * self.descuento_porcentaje) / 100
        
        self.total = self.subtotal - self.descuento_monto
    
    @property
    def total_pagado(self):
        """Total pagado en esta venta"""
        return sum(pago.monto for pago in self.pagos.filter(activo=True))
    
    @property
    def saldo_pendiente(self):
        """Saldo pendiente de pago"""
        return self.total - self.total_pagado
    
    @property
    def esta_pagada(self):
        """Verificar si la venta está completamente pagada"""
        return self.saldo_pendiente <= 0
    
    def puede_ser_entregada(self):
        """Verificar si puede ser entregada"""
        return self.estado == 'lista' and self.esta_pagada


class ItemVenta(models.Model):
    """Items individuales de una venta"""
    
    venta = models.ForeignKey(
        Venta,
        on_delete=models.CASCADE,
        related_name='items'
    )
    producto = models.ForeignKey(
        Producto,
        on_delete=models.PROTECT
    )
    cantidad = models.IntegerField(
        default=1,
        validators=[MinValueValidator(1)],
        verbose_name='Cantidad'
    )
    precio_unitario = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Precio Unitario'
    )
    subtotal = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Subtotal'
    )
    observaciones = models.TextField(
        blank=True,
        null=True,
        verbose_name='Observaciones'
    )
    
    class Meta:
        verbose_name = 'Item de Venta'
        verbose_name_plural = 'Items de Venta'
    
    def __str__(self):
        return f"{self.producto.nombre} x{self.cantidad}"
    
    def save(self, *args, **kwargs):
        # Calcular subtotal
        self.subtotal = self.cantidad * self.precio_unitario
        super().save(*args, **kwargs)
        
        # Actualizar totales de la venta
        self.venta.calcular_totales()
        self.venta.save()


class Pago(models.Model):
    """Pagos realizados por las ventas"""
    
    METODO_CHOICES = [
        ('efectivo', 'Efectivo'),
        ('transferencia', 'Transferencia'),
        ('tarjeta', 'Tarjeta'),
        ('cheque', 'Cheque'),
    ]
    
    venta = models.ForeignKey(
        Venta,
        on_delete=models.CASCADE,
        related_name='pagos'
    )
    fecha_pago = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Pago'
    )
    metodo_pago = models.CharField(
        max_length=20,
        choices=METODO_CHOICES,
        verbose_name='Método de Pago'
    )
    moneda = models.CharField(
        max_length=3,
        choices=[
            ('CUP', 'Peso Cubano'),
            ('USD', 'Dólar Estadounidense'),
            ('EUR', 'Euro'),
            ('MLC', 'Moneda Libremente Convertible'),
        ],
        verbose_name='Moneda'
    )
    monto = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Monto'
    )
    monto_cup = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Monto en CUP'
    )
    tasa_cambio = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=1,
        validators=[MinValueValidator(0)],
        verbose_name='Tasa de Cambio'
    )
    referencia = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name='Referencia'
    )
    observaciones = models.TextField(
        blank=True,
        null=True,
        verbose_name='Observaciones'
    )
    recibido_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Recibido por'
    )
    activo = models.BooleanField(
        default=True,
        verbose_name='Activo'
    )
    
    class Meta:
        verbose_name = 'Pago'
        verbose_name_plural = 'Pagos'
        ordering = ['-fecha_pago']
    
    def __str__(self):
        return f"Pago {self.monto} {self.moneda} - Factura {self.venta.numero_factura}"
    
    def save(self, *args, **kwargs):
        # Convertir a CUP si es necesario
        if self.moneda == 'CUP':
            self.monto_cup = self.monto
            self.tasa_cambio = 1
        else:
            # Obtener tasa de cambio actual
            try:
                tasa = TasaCambio.objects.get(moneda=self.moneda)
                self.tasa_cambio = tasa.tasa_cup
                self.monto_cup = self.monto * self.tasa_cambio
            except TasaCambio.DoesNotExist:
                self.monto_cup = self.monto
        
        super().save(*args, **kwargs)


class Descuento(models.Model):
    """Descuentos aplicables a las ventas"""
    
    TIPO_CHOICES = [
        ('porcentaje', 'Porcentaje'),
        ('monto_fijo', 'Monto Fijo'),
    ]
    
    nombre = models.CharField(
        max_length=100,
        verbose_name='Nombre'
    )
    descripcion = models.TextField(
        blank=True,
        null=True,
        verbose_name='Descripción'
    )
    tipo = models.CharField(
        max_length=20,
        choices=TIPO_CHOICES,
        verbose_name='Tipo'
    )
    valor = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Valor'
    )
    monto_minimo = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        verbose_name='Monto Mínimo'
    )
    fecha_inicio = models.DateField(
        verbose_name='Fecha de Inicio'
    )
    fecha_fin = models.DateField(
        verbose_name='Fecha de Fin'
    )
    activo = models.BooleanField(
        default=True,
        verbose_name='Activo'
    )
    
    class Meta:
        verbose_name = 'Descuento'
        verbose_name_plural = 'Descuentos'
        ordering = ['nombre']
    
    def __str__(self):
        if self.tipo == 'porcentaje':
            return f"{self.nombre} ({self.valor}%)"
        else:
            return f"{self.nombre} ({self.valor} CUP)"
    
    def calcular_descuento(self, monto):
        """Calcular el descuento para un monto dado"""
        if monto < self.monto_minimo:
            return 0
        
        if self.tipo == 'porcentaje':
            return (monto * self.valor) / 100
        else:
            return self.valor


class Devolucion(models.Model):
    """Devoluciones de productos"""
    
    MOTIVO_CHOICES = [
        ('defecto', 'Defecto de Fábrica'),
        ('no_satisfecho', 'Cliente No Satisfecho'),
        ('error_graduacion', 'Error en Graduación'),
        ('cambio_modelo', 'Cambio de Modelo'),
        ('otro', 'Otro'),
    ]
    
    venta = models.ForeignKey(
        Venta,
        on_delete=models.CASCADE,
        related_name='devoluciones'
    )
    fecha_devolucion = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Devolución'
    )
    motivo = models.CharField(
        max_length=20,
        choices=MOTIVO_CHOICES,
        verbose_name='Motivo'
    )
    descripcion = models.TextField(
        verbose_name='Descripción'
    )
    monto_devuelto = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Monto Devuelto'
    )
    autorizada_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Autorizada por'
    )
    
    class Meta:
        verbose_name = 'Devolución'
        verbose_name_plural = 'Devoluciones'
        ordering = ['-fecha_devolucion']
    
    def __str__(self):
        return f"Devolución - Factura {self.venta.numero_factura}"
