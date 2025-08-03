from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Categoria(models.Model):
    nombre = models.CharField(max_length=100, unique=True)
    descripcion = models.TextField(blank=True)
    activa = models.BooleanField(default=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Categoría"
        verbose_name_plural = "Categorías"
        ordering = ['nombre']
    
    def __str__(self):
        return self.nombre

class Producto(models.Model):
    MONEDA_CHOICES = [
        ('CUP', 'Peso Cubano'),
        ('USD', 'Dólar Americano'),
        ('EUR', 'Euro'),
        ('MLC', 'Moneda Libremente Convertible'),
    ]
    
    ESTADO_CHOICES = [
        ('activo', 'Activo'),
        ('inactivo', 'Inactivo'),
        ('agotado', 'Agotado'),
    ]
    
    # Información básica
    nombre = models.CharField(max_length=200, verbose_name="Nombre del Producto")
    descripcion = models.TextField(verbose_name="Descripción")
    codigo = models.CharField(max_length=50, unique=True, verbose_name="Código del Producto")
    
    # Precios
    precio_compra = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="Precio de Compra")
    precio_venta = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="Precio de Venta")
    moneda = models.CharField(max_length=3, choices=MONEDA_CHOICES, default='CUP', verbose_name="Moneda")
    
    # Inventario
    stock = models.PositiveIntegerField(default=0, verbose_name="Stock Disponible")
    stock_minimo = models.PositiveIntegerField(default=5, verbose_name="Stock Mínimo")
    
    # Categorización
    categoria = models.ForeignKey(Categoria, on_delete=models.CASCADE, verbose_name="Categoría")
    
    # Imágenes
    imagen_principal = models.ImageField(upload_to='productos/', blank=True, null=True, verbose_name="Imagen Principal")
    imagen_2 = models.ImageField(upload_to='productos/', blank=True, null=True, verbose_name="Imagen 2")
    imagen_3 = models.ImageField(upload_to='productos/', blank=True, null=True, verbose_name="Imagen 3")
    
    # Estado y fechas
    estado = models.CharField(max_length=10, choices=ESTADO_CHOICES, default='activo')
    destacado = models.BooleanField(default=False, verbose_name="Producto Destacado")
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_modificacion = models.DateTimeField(auto_now=True)
    
    # Métricas
    vistas = models.PositiveIntegerField(default=0)
    ventas_totales = models.PositiveIntegerField(default=0)
    
    class Meta:
        verbose_name = "Producto"
        verbose_name_plural = "Productos"
        ordering = ['-fecha_creacion']
        indexes = [
            models.Index(fields=['codigo']),
            models.Index(fields=['categoria']),
            models.Index(fields=['estado']),
        ]
    
    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    
    def get_absolute_url(self):
        return reverse('producto_detalle', kwargs={'pk': self.pk})
    
    @property
    def margen_ganancia(self):
        """Calcula el margen de ganancia en porcentaje"""
        if self.precio_compra > 0:
            return ((self.precio_venta - self.precio_compra) / self.precio_compra) * 100
        return 0
    
    @property
    def esta_agotado(self):
        """Verifica si el producto está agotado"""
        return self.stock <= 0
    
    @property
    def stock_bajo(self):
        """Verifica si el stock está por debajo del mínimo"""
        return self.stock <= self.stock_minimo
    
    def incrementar_vistas(self):
        """Incrementa el contador de vistas"""
        self.vistas += 1
        self.save(update_fields=['vistas'])

class ImagenProducto(models.Model):
    producto = models.ForeignKey(Producto, related_name='imagenes_adicionales', on_delete=models.CASCADE)
    imagen = models.ImageField(upload_to='productos/adicionales/')
    descripcion = models.CharField(max_length=200, blank=True)
    orden = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['orden']
        verbose_name = "Imagen Adicional"
        verbose_name_plural = "Imágenes Adicionales"
    
    def __str__(self):
        return f"Imagen de {self.producto.nombre}"
