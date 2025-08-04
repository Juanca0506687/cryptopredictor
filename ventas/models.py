from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from decimal import Decimal
import uuid
from django.utils import timezone

class Venta(models.Model):
    ESTADOS = [
        ('borrador', 'Borrador'),
        ('confirmada', 'Confirmada'),
        ('en_taller', 'En Taller'),
        ('lista', 'Lista para Entrega'),
        ('entregada', 'Entregada'),
        ('cancelada', 'Cancelada'),
    ]
    
    MONEDAS = [
        ('CUP', 'Pesos Cubanos'),
        ('USD', 'Dólares Americanos'),
        ('EUR', 'Euros'),
        ('MLC', 'Moneda Libremente Convertible'),
    ]
    
    # Identificación
    numero_factura = models.CharField(max_length=20, unique=True, editable=False)
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    
    # Relaciones
    cliente = models.ForeignKey('clientes.Cliente', on_delete=models.CASCADE)
    vendedor = models.ForeignKey('usuarios.Usuario', on_delete=models.CASCADE, related_name='ventas_realizadas')
    receta = models.ForeignKey('clientes.RecetaMedica', on_delete=models.SET_NULL, blank=True, null=True)
    
    # Fechas
    fecha = models.DateTimeField(auto_now_add=True)
    fecha_entrega_estimada = models.DateField(blank=True, null=True)
    fecha_entrega_real = models.DateTimeField(blank=True, null=True)
    
    # Montos
    subtotal = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    descuento = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    impuestos = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    moneda = models.CharField(max_length=3, choices=MONEDAS, default='CUP')
    
    # Estado y control
    estado = models.CharField(max_length=20, choices=ESTADOS, default='borrador')
    observaciones = models.TextField(blank=True, null=True)
    
    # Control
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Venta'
        verbose_name_plural = 'Ventas'
        ordering = ['-fecha']
    
    def __str__(self):
        return f"Factura {self.numero_factura} - {self.cliente.nombre_completo}"
    
    def save(self, *args, **kwargs):
        if not self.numero_factura:
            # Generar número de factura automático
            ultimo_numero = Venta.objects.filter(
                fecha__year=self.fecha.year if self.fecha else timezone.now().year
            ).count() + 1
            year = self.fecha.year if self.fecha else timezone.now().year
            self.numero_factura = f"F{year}{ultimo_numero:04d}"
        
        # Calcular totales
        self.calcular_totales()
        super().save(*args, **kwargs)
    
    def calcular_totales(self):
        """Calcula los totales de la venta"""
        items = self.items.all()
        self.subtotal = sum(item.subtotal for item in items)
        self.total = self.subtotal - self.descuento + self.impuestos
    
    @property
    def total_pagado(self):
        return sum(pago.monto for pago in self.pagos.filter(activo=True))
    
    @property
    def saldo_pendiente(self):
        return self.total - self.total_pagado
    
    @property
    def esta_pagada(self):
        return self.saldo_pendiente <= 0
    
    def puede_entregar(self):
        return self.estado == 'lista' and self.esta_pagada

class ItemVenta(models.Model):
    venta = models.ForeignKey(Venta, on_delete=models.CASCADE, related_name='items')
    
    # Producto (puede ser montura o cristal)
    montura = models.ForeignKey('productos.Montura', on_delete=models.CASCADE, blank=True, null=True)
    cristal = models.ForeignKey('productos.Cristal', on_delete=models.CASCADE, blank=True, null=True)
    
    # Detalles del item
    descripcion = models.CharField(max_length=200)
    cantidad = models.IntegerField(default=1, validators=[MinValueValidator(1)])
    precio_unitario = models.DecimalField(max_digits=10, decimal_places=2)
    descuento_item = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Graduación específica para cristales
    od_esfera = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, blank=True, null=True)
    od_cilindro = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, blank=True, null=True)
    od_eje = models.IntegerField(default=0, blank=True, null=True)
    oi_esfera = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, blank=True, null=True)
    oi_cilindro = models.DecimalField(max_digits=4, decimal_places=2, default=0.00, blank=True, null=True)
    oi_eje = models.IntegerField(default=0, blank=True, null=True)
    
    class Meta:
        verbose_name = 'Item de Venta'
        verbose_name_plural = 'Items de Venta'
    
    def __str__(self):
        return f"{self.descripcion} - {self.cantidad} x {self.precio_unitario}"
    
    @property
    def subtotal(self):
        return (self.cantidad * self.precio_unitario) - self.descuento_item
    
    def clean(self):
        from django.core.exceptions import ValidationError
        if not self.montura and not self.cristal:
            raise ValidationError('Debe especificar una montura o un cristal')
        if self.montura and self.cristal:
            raise ValidationError('No puede especificar tanto montura como cristal')

class Pago(models.Model):
    METODOS_PAGO = [
        ('efectivo', 'Efectivo'),
        ('transferencia', 'Transferencia'),
        ('tarjeta', 'Tarjeta'),
        ('cheque', 'Cheque'),
        ('credito', 'Crédito'),
    ]
    
    MONEDAS = [
        ('CUP', 'Pesos Cubanos'),
        ('USD', 'Dólares Americanos'),
        ('EUR', 'Euros'),
        ('MLC', 'Moneda Libremente Convertible'),
    ]
    
    venta = models.ForeignKey(Venta, on_delete=models.CASCADE, related_name='pagos')
    fecha_pago = models.DateTimeField(auto_now_add=True)
    metodo_pago = models.CharField(max_length=20, choices=METODOS_PAGO)
    monto = models.DecimalField(max_digits=12, decimal_places=2, validators=[MinValueValidator(Decimal('0.01'))])
    moneda = models.CharField(max_length=3, choices=MONEDAS, default='CUP')
    
    # Información adicional según método de pago
    numero_referencia = models.CharField(max_length=100, blank=True, null=True)
    banco = models.CharField(max_length=100, blank=True, null=True)
    observaciones = models.TextField(blank=True, null=True)
    
    # Control
    activo = models.BooleanField(default=True)
    usuario_registro = models.ForeignKey('usuarios.Usuario', on_delete=models.CASCADE)
    
    class Meta:
        verbose_name = 'Pago'
        verbose_name_plural = 'Pagos'
        ordering = ['-fecha_pago']
    
    def __str__(self):
        return f"Pago {self.monto} {self.moneda} - {self.venta.numero_factura}"

class TasaCambio(models.Model):
    moneda_origen = models.CharField(max_length=3, choices=[
        ('CUP', 'Pesos Cubanos'),
        ('USD', 'Dólares Americanos'),
        ('EUR', 'Euros'),
        ('MLC', 'Moneda Libremente Convertible'),
    ])
    moneda_destino = models.CharField(max_length=3, choices=[
        ('CUP', 'Pesos Cubanos'),
        ('USD', 'Dólares Americanos'),
        ('EUR', 'Euros'),
        ('MLC', 'Moneda Libremente Convertible'),
    ])
    tasa = models.DecimalField(max_digits=10, decimal_places=4, validators=[MinValueValidator(Decimal('0.0001'))])
    fecha_vigencia = models.DateTimeField(auto_now_add=True)
    activa = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = 'Tasa de Cambio'
        verbose_name_plural = 'Tasas de Cambio'
        unique_together = ['moneda_origen', 'moneda_destino', 'fecha_vigencia']
        ordering = ['-fecha_vigencia']
    
    def __str__(self):
        return f"{self.moneda_origen} -> {self.moneda_destino}: {self.tasa}"
    
    @classmethod
    def convertir(cls, monto, moneda_origen, moneda_destino):
        """Convierte un monto de una moneda a otra"""
        if moneda_origen == moneda_destino:
            return monto
        
        tasa = cls.objects.filter(
            moneda_origen=moneda_origen,
            moneda_destino=moneda_destino,
            activa=True
        ).first()
        
        if tasa:
            return monto * tasa.tasa
        return monto  # Si no hay tasa, devolver el monto original

class Descuento(models.Model):
    TIPOS_DESCUENTO = [
        ('porcentaje', 'Porcentaje'),
        ('monto_fijo', 'Monto Fijo'),
    ]
    
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField(blank=True, null=True)
    tipo = models.CharField(max_length=20, choices=TIPOS_DESCUENTO)
    valor = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Condiciones
    monto_minimo = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    cantidad_minima = models.IntegerField(default=1)
    
    # Vigencia
    fecha_inicio = models.DateField()
    fecha_fin = models.DateField()
    activo = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = 'Descuento'
        verbose_name_plural = 'Descuentos'
        ordering = ['nombre']
    
    def __str__(self):
        return f"{self.nombre} ({self.valor}{'%' if self.tipo == 'porcentaje' else ' CUP'})"
    
    def calcular_descuento(self, monto):
        """Calcula el descuento aplicable a un monto"""
        if self.tipo == 'porcentaje':
            return monto * (self.valor / 100)
        else:
            return self.valor
    
    def es_aplicable(self, monto, cantidad=1):
        """Verifica si el descuento es aplicable"""
        from datetime import date
        today = date.today()
        return (self.activo and 
                self.fecha_inicio <= today <= self.fecha_fin and
                monto >= self.monto_minimo and
                cantidad >= self.cantidad_minima)
