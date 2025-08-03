from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Categoria(models.Model):
    nombre = models.CharField(max_length=100, unique=True)
    descripcion = models.TextField(blank=True)
    imagen = models.ImageField(upload_to='categorias/', blank=True, null=True)
    activa = models.BooleanField(default=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Categoría'
        verbose_name_plural = 'Categorías'
        ordering = ['nombre']
    
    def __str__(self):
        return self.nombre

class Producto(models.Model):
    MONEDAS = [
        ('CUP', 'Peso Cubano (CUP)'),
        ('USD', 'Dólar Americano (USD)'),
        ('EUR', 'Euro (EUR)'),
    ]
    
    ESTADOS = [
        ('disponible', 'Disponible'),
        ('agotado', 'Agotado'),
        ('descontinuado', 'Descontinuado'),
    ]
    
    # Información básica
    nombre = models.CharField(max_length=200, verbose_name='Nombre del producto')
    descripcion = models.TextField(verbose_name='Descripción')
    codigo = models.CharField(max_length=50, unique=True, verbose_name='Código del producto')
    categoria = models.ForeignKey(Categoria, on_delete=models.CASCADE, related_name='productos')
    
    # Precios
    precio_compra = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='Precio de compra')
    precio_venta = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='Precio de venta')
    moneda = models.CharField(max_length=3, choices=MONEDAS, default='CUP', verbose_name='Moneda')
    
    # Inventario
    stock = models.PositiveIntegerField(default=0, verbose_name='Cantidad en stock')
    stock_minimo = models.PositiveIntegerField(default=5, verbose_name='Stock mínimo')
    estado = models.CharField(max_length=20, choices=ESTADOS, default='disponible')
    
    # Imágenes
    imagen_principal = models.ImageField(upload_to='productos/', verbose_name='Imagen principal')
    imagen_2 = models.ImageField(upload_to='productos/', blank=True, null=True, verbose_name='Imagen 2')
    imagen_3 = models.ImageField(upload_to='productos/', blank=True, null=True, verbose_name='Imagen 3')
    
    # Características adicionales
    peso = models.DecimalField(max_digits=8, decimal_places=2, blank=True, null=True, verbose_name='Peso (kg)')
    dimensiones = models.CharField(max_length=100, blank=True, verbose_name='Dimensiones')
    marca = models.CharField(max_length=100, blank=True, verbose_name='Marca')
    
    # Metadata
    activo = models.BooleanField(default=True)
    destacado = models.BooleanField(default=False, verbose_name='Producto destacado')
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
    creado_por = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    class Meta:
        verbose_name = 'Producto'
        verbose_name_plural = 'Productos'
        ordering = ['-fecha_creacion']
    
    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    
    def get_absolute_url(self):
        return reverse('producto_detalle', kwargs={'pk': self.pk})
    
    @property
    def margen_ganancia(self):
        """Calcula el margen de ganancia"""
        if self.precio_compra > 0:
            return ((self.precio_venta - self.precio_compra) / self.precio_compra) * 100
        return 0
    
    @property
    def disponible(self):
        """Verifica si el producto está disponible"""
        return self.activo and self.estado == 'disponible' and self.stock > 0
    
    @property
    def stock_bajo(self):
        """Verifica si el stock está bajo"""
        return self.stock <= self.stock_minimo

class ImagenProducto(models.Model):
    producto = models.ForeignKey(Producto, on_delete=models.CASCADE, related_name='imagenes_adicionales')
    imagen = models.ImageField(upload_to='productos/adicionales/')
    descripcion = models.CharField(max_length=200, blank=True)
    orden = models.PositiveIntegerField(default=0)
    
    class Meta:
        verbose_name = 'Imagen adicional'
        verbose_name_plural = 'Imágenes adicionales'
        ordering = ['orden']
    
    def __str__(self):
        return f"Imagen de {self.producto.nombre}"