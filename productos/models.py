from django.db import models
from django.core.validators import MinValueValidator
from django.contrib.auth import get_user_model

Usuario = get_user_model()


class CategoriaProducto(models.Model):
    """Categorías de productos"""
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
    activa = models.BooleanField(
        default=True,
        verbose_name='Activa'
    )
    
    class Meta:
        verbose_name = 'Categoría de Producto'
        verbose_name_plural = 'Categorías de Productos'
        ordering = ['nombre']
    
    def __str__(self):
        return self.nombre


class Marca(models.Model):
    """Marcas de productos"""
    nombre = models.CharField(
        max_length=100,
        unique=True,
        verbose_name='Nombre'
    )
    pais_origen = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        verbose_name='País de Origen'
    )
    activa = models.BooleanField(
        default=True,
        verbose_name='Activa'
    )
    
    class Meta:
        verbose_name = 'Marca'
        verbose_name_plural = 'Marcas'
        ordering = ['nombre']
    
    def __str__(self):
        return self.nombre


class Producto(models.Model):
    """Modelo base para productos"""
    
    TIPO_CHOICES = [
        ('montura', 'Montura'),
        ('cristal', 'Cristal'),
        ('accesorio', 'Accesorio'),
        ('servicio', 'Servicio'),
    ]
    
    codigo = models.CharField(
        max_length=50,
        unique=True,
        verbose_name='Código'
    )
    nombre = models.CharField(
        max_length=200,
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
    categoria = models.ForeignKey(
        CategoriaProducto,
        on_delete=models.PROTECT,
        verbose_name='Categoría'
    )
    marca = models.ForeignKey(
        Marca,
        on_delete=models.PROTECT,
        blank=True,
        null=True,
        verbose_name='Marca'
    )
    precio_costo = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Precio de Costo'
    )
    precio_venta_cup = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        verbose_name='Precio de Venta (CUP)'
    )
    precio_venta_usd = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        blank=True,
        null=True,
        verbose_name='Precio de Venta (USD)'
    )
    precio_venta_eur = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        blank=True,
        null=True,
        verbose_name='Precio de Venta (EUR)'
    )
    precio_venta_mlc = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        blank=True,
        null=True,
        verbose_name='Precio de Venta (MLC)'
    )
    stock_minimo = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        verbose_name='Stock Mínimo'
    )
    activo = models.BooleanField(
        default=True,
        verbose_name='Activo'
    )
    fecha_creacion = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Creación'
    )
    fecha_actualizacion = models.DateTimeField(
        auto_now=True,
        verbose_name='Fecha de Actualización'
    )
    creado_por = models.ForeignKey(
        Usuario,
        on_delete=models.PROTECT,
        verbose_name='Creado por'
    )
    
    class Meta:
        verbose_name = 'Producto'
        verbose_name_plural = 'Productos'
        ordering = ['nombre']
    
    def __str__(self):
        return f"{self.codigo} - {self.nombre}"
    
    @property
    def stock_actual(self):
        """Calcular stock actual basado en movimientos"""
        return self.movimientos.filter(activo=True).aggregate(
            total=models.Sum('cantidad')
        )['total'] or 0
    
    @property
    def necesita_reposicion(self):
        """Verificar si necesita reposición"""
        return self.stock_actual <= self.stock_minimo
    
    def get_precio_por_moneda(self, moneda):
        """Obtener precio según la moneda"""
        precios = {
            'CUP': self.precio_venta_cup,
            'USD': self.precio_venta_usd,
            'EUR': self.precio_venta_eur,
            'MLC': self.precio_venta_mlc,
        }
        return precios.get(moneda, self.precio_venta_cup)


class Montura(models.Model):
    """Modelo específico para monturas"""
    producto = models.OneToOneField(
        Producto,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='montura'
    )
    
    MATERIAL_CHOICES = [
        ('metal', 'Metal'),
        ('plastico', 'Plástico'),
        ('titanio', 'Titanio'),
        ('acetato', 'Acetato'),
        ('mixto', 'Mixto'),
    ]
    
    GENERO_CHOICES = [
        ('unisex', 'Unisex'),
        ('hombre', 'Hombre'),
        ('mujer', 'Mujer'),
        ('nino', 'Niño'),
    ]
    
    material = models.CharField(
        max_length=20,
        choices=MATERIAL_CHOICES,
        verbose_name='Material'
    )
    color = models.CharField(
        max_length=50,
        verbose_name='Color'
    )
    genero = models.CharField(
        max_length=10,
        choices=GENERO_CHOICES,
        verbose_name='Género'
    )
    ancho_lente = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='Ancho del Lente (mm)'
    )
    alto_lente = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='Alto del Lente (mm)'
    )
    puente = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='Puente (mm)'
    )
    varilla = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='Varilla (mm)'
    )
    
    class Meta:
        verbose_name = 'Montura'
        verbose_name_plural = 'Monturas'
    
    def __str__(self):
        return f"Montura: {self.producto.nombre}"


class Cristal(models.Model):
    """Modelo específico para cristales"""
    producto = models.OneToOneField(
        Producto,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='cristal'
    )
    
    TIPO_CHOICES = [
        ('monofocal', 'Monofocal'),
        ('bifocal', 'Bifocal'),
        ('progresivo', 'Progresivo'),
        ('ocupacional', 'Ocupacional'),
    ]
    
    MATERIAL_CHOICES = [
        ('organico', 'Orgánico'),
        ('policarbonato', 'Policarbonato'),
        ('trivex', 'Trivex'),
        ('mineral', 'Mineral'),
    ]
    
    tipo = models.CharField(
        max_length=20,
        choices=TIPO_CHOICES,
        verbose_name='Tipo'
    )
    material = models.CharField(
        max_length=20,
        choices=MATERIAL_CHOICES,
        verbose_name='Material'
    )
    indice_refraccion = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        blank=True,
        null=True,
        verbose_name='Índice de Refracción'
    )
    anti_reflejo = models.BooleanField(
        default=False,
        verbose_name='Anti-reflejo'
    )
    fotocromatico = models.BooleanField(
        default=False,
        verbose_name='Fotocromático'
    )
    polarizado = models.BooleanField(
        default=False,
        verbose_name='Polarizado'
    )
    filtro_azul = models.BooleanField(
        default=False,
        verbose_name='Filtro Luz Azul'
    )
    
    class Meta:
        verbose_name = 'Cristal'
        verbose_name_plural = 'Cristales'
    
    def __str__(self):
        return f"Cristal: {self.producto.nombre}"


class MovimientoInventario(models.Model):
    """Movimientos de inventario"""
    
    TIPO_CHOICES = [
        ('entrada', 'Entrada'),
        ('salida', 'Salida'),
        ('ajuste', 'Ajuste'),
        ('devolucion', 'Devolución'),
    ]
    
    producto = models.ForeignKey(
        Producto,
        on_delete=models.CASCADE,
        related_name='movimientos'
    )
    tipo = models.CharField(
        max_length=20,
        choices=TIPO_CHOICES,
        verbose_name='Tipo'
    )
    cantidad = models.IntegerField(
        verbose_name='Cantidad'
    )
    motivo = models.CharField(
        max_length=200,
        verbose_name='Motivo'
    )
    observaciones = models.TextField(
        blank=True,
        null=True,
        verbose_name='Observaciones'
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
    activo = models.BooleanField(
        default=True,
        verbose_name='Activo'
    )
    
    class Meta:
        verbose_name = 'Movimiento de Inventario'
        verbose_name_plural = 'Movimientos de Inventario'
        ordering = ['-fecha']
    
    def __str__(self):
        return f"{self.tipo.title()} - {self.producto.nombre} ({self.cantidad})"
    
    def save(self, *args, **kwargs):
        # Convertir salidas a números negativos
        if self.tipo == 'salida' and self.cantidad > 0:
            self.cantidad = -self.cantidad
        super().save(*args, **kwargs)


class AlertaStock(models.Model):
    """Alertas de stock bajo"""
    producto = models.ForeignKey(
        Producto,
        on_delete=models.CASCADE,
        related_name='alertas'
    )
    mensaje = models.CharField(
        max_length=200,
        verbose_name='Mensaje'
    )
    fecha_creacion = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de Creación'
    )
    vista = models.BooleanField(
        default=False,
        verbose_name='Vista'
    )
    resuelta = models.BooleanField(
        default=False,
        verbose_name='Resuelta'
    )
    
    class Meta:
        verbose_name = 'Alerta de Stock'
        verbose_name_plural = 'Alertas de Stock'
        ordering = ['-fecha_creacion']
    
    def __str__(self):
        return f"Alerta: {self.producto.nombre}"
