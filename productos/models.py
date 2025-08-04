from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from PIL import Image
import os

class Categoria(models.Model):
    nombre = models.CharField(max_length=100, unique=True)
    descripcion = models.TextField(blank=True, null=True)
    activo = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = 'Categoría'
        verbose_name_plural = 'Categorías'
    
    def __str__(self):
        return self.nombre

class Proveedor(models.Model):
    nombre = models.CharField(max_length=200)
    contacto = models.CharField(max_length=100, blank=True, null=True)
    telefono = models.CharField(max_length=15, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    direccion = models.TextField(blank=True, null=True)
    activo = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = 'Proveedor'
        verbose_name_plural = 'Proveedores'
    
    def __str__(self):
        return self.nombre

class Montura(models.Model):
    MATERIALES = [
        ('metal', 'Metal'),
        ('plastico', 'Plástico'),
        ('titanio', 'Titanio'),
        ('acetato', 'Acetato'),
        ('mixto', 'Mixto'),
    ]
    
    GENEROS = [
        ('unisex', 'Unisex'),
        ('hombre', 'Hombre'),
        ('mujer', 'Mujer'),
        ('niño', 'Niño'),
    ]
    
    codigo = models.CharField(max_length=50, unique=True)
    nombre = models.CharField(max_length=200)
    categoria = models.ForeignKey(Categoria, on_delete=models.CASCADE)
    proveedor = models.ForeignKey(Proveedor, on_delete=models.CASCADE)
    material = models.CharField(max_length=20, choices=MATERIALES)
    genero = models.CharField(max_length=10, choices=GENEROS, default='unisex')
    color = models.CharField(max_length=50)
    talla = models.CharField(max_length=20, blank=True, null=True)
    precio_compra = models.DecimalField(max_digits=10, decimal_places=2)
    precio_venta = models.DecimalField(max_digits=10, decimal_places=2)
    stock_actual = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    stock_minimo = models.IntegerField(default=5, validators=[MinValueValidator(0)])
    imagen = models.ImageField(upload_to='monturas/', blank=True, null=True)
    descripcion = models.TextField(blank=True, null=True)
    activo = models.BooleanField(default=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Montura'
        verbose_name_plural = 'Monturas'
        ordering = ['nombre']
    
    def __str__(self):
        return f"{self.codigo} - {self.nombre}"
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.imagen:
            img = Image.open(self.imagen.path)
            if img.height > 300 or img.width > 300:
                output_size = (300, 300)
                img.thumbnail(output_size)
                img.save(self.imagen.path)
    
    @property
    def necesita_restock(self):
        return self.stock_actual <= self.stock_minimo
    
    @property
    def margen_ganancia(self):
        if self.precio_compra > 0:
            return ((self.precio_venta - self.precio_compra) / self.precio_compra) * 100
        return 0

class TipoCristal(models.Model):
    nombre = models.CharField(max_length=100, unique=True)
    descripcion = models.TextField(blank=True, null=True)
    precio_base = models.DecimalField(max_digits=10, decimal_places=2)
    activo = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = 'Tipo de Cristal'
        verbose_name_plural = 'Tipos de Cristales'
    
    def __str__(self):
        return self.nombre

class Cristal(models.Model):
    TIPOS_GRADUACION = [
        ('monofocal', 'Monofocal'),
        ('bifocal', 'Bifocal'),
        ('progresivo', 'Progresivo'),
        ('ocupacional', 'Ocupacional'),
    ]
    
    codigo = models.CharField(max_length=50, unique=True)
    tipo_cristal = models.ForeignKey(TipoCristal, on_delete=models.CASCADE)
    tipo_graduacion = models.CharField(max_length=20, choices=TIPOS_GRADUACION)
    proveedor = models.ForeignKey(Proveedor, on_delete=models.CASCADE)
    
    # Graduación
    esfera_min = models.DecimalField(max_digits=4, decimal_places=2, default=-20.00)
    esfera_max = models.DecimalField(max_digits=4, decimal_places=2, default=20.00)
    cilindro_min = models.DecimalField(max_digits=4, decimal_places=2, default=-6.00)
    cilindro_max = models.DecimalField(max_digits=4, decimal_places=2, default=6.00)
    
    precio_compra = models.DecimalField(max_digits=10, decimal_places=2)
    precio_venta = models.DecimalField(max_digits=10, decimal_places=2)
    stock_actual = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    stock_minimo = models.IntegerField(default=2, validators=[MinValueValidator(0)])
    
    # Tratamientos adicionales
    antirreflejo = models.BooleanField(default=False)
    fotocromatico = models.BooleanField(default=False)
    polarizado = models.BooleanField(default=False)
    
    descripcion = models.TextField(blank=True, null=True)
    activo = models.BooleanField(default=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Cristal'
        verbose_name_plural = 'Cristales'
        ordering = ['tipo_cristal', 'tipo_graduacion']
    
    def __str__(self):
        return f"{self.codigo} - {self.tipo_cristal.nombre} ({self.get_tipo_graduacion_display()})"
    
    @property
    def necesita_restock(self):
        return self.stock_actual <= self.stock_minimo
    
    def puede_graduar(self, esfera, cilindro):
        """Verifica si el cristal puede manejar la graduación especificada"""
        return (self.esfera_min <= esfera <= self.esfera_max and 
                self.cilindro_min <= cilindro <= self.cilindro_max)

class MovimientoInventario(models.Model):
    TIPOS_MOVIMIENTO = [
        ('entrada', 'Entrada'),
        ('salida', 'Salida'),
        ('ajuste', 'Ajuste'),
        ('devolucion', 'Devolución'),
    ]
    
    fecha = models.DateTimeField(auto_now_add=True)
    tipo_movimiento = models.CharField(max_length=20, choices=TIPOS_MOVIMIENTO)
    
    # Producto (puede ser montura o cristal)
    montura = models.ForeignKey(Montura, on_delete=models.CASCADE, blank=True, null=True)
    cristal = models.ForeignKey(Cristal, on_delete=models.CASCADE, blank=True, null=True)
    
    cantidad = models.IntegerField(validators=[MinValueValidator(1)])
    precio_unitario = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    motivo = models.CharField(max_length=200)
    observaciones = models.TextField(blank=True, null=True)
    
    # Usuario que realiza el movimiento
    usuario = models.ForeignKey('usuarios.Usuario', on_delete=models.CASCADE)
    
    class Meta:
        verbose_name = 'Movimiento de Inventario'
        verbose_name_plural = 'Movimientos de Inventario'
        ordering = ['-fecha']
    
    def __str__(self):
        producto = self.montura or self.cristal
        return f"{self.get_tipo_movimiento_display()} - {producto} - {self.cantidad}"
    
    def clean(self):
        from django.core.exceptions import ValidationError
        if not self.montura and not self.cristal:
            raise ValidationError('Debe especificar una montura o un cristal')
        if self.montura and self.cristal:
            raise ValidationError('No puede especificar tanto montura como cristal')
    
    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)
        
        # Actualizar stock
        if self.montura:
            if self.tipo_movimiento in ['entrada', 'ajuste']:
                self.montura.stock_actual += self.cantidad
            else:  # salida, devolucion
                self.montura.stock_actual -= self.cantidad
            self.montura.save()
        
        if self.cristal:
            if self.tipo_movimiento in ['entrada', 'ajuste']:
                self.cristal.stock_actual += self.cantidad
            else:  # salida, devolucion
                self.cristal.stock_actual -= self.cantidad
            self.cristal.save()
