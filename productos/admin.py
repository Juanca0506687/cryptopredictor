from django.contrib import admin
from django.utils.html import format_html
from .models import Categoria, Producto, ImagenProducto

@admin.register(Categoria)
class CategoriaAdmin(admin.ModelAdmin):
    list_display = ['nombre', 'activa', 'fecha_creacion', 'cantidad_productos']
    list_filter = ['activa', 'fecha_creacion']
    search_fields = ['nombre', 'descripcion']
    prepopulated_fields = {'nombre': ('nombre',)}
    
    def cantidad_productos(self, obj):
        return obj.productos.count()
    cantidad_productos.short_description = 'Productos'

class ImagenProductoInline(admin.TabularInline):
    model = ImagenProducto
    extra = 1
    fields = ['imagen', 'descripcion', 'orden']

@admin.register(Producto)
class ProductoAdmin(admin.ModelAdmin):
    list_display = [
        'codigo', 'nombre', 'categoria', 'precio_venta_formateado', 
        'moneda', 'stock', 'estado', 'stock_status', 'activo'
    ]
    list_filter = ['categoria', 'moneda', 'estado', 'activo', 'destacado', 'fecha_creacion']
    search_fields = ['nombre', 'codigo', 'descripcion', 'marca']
    list_editable = ['precio_venta', 'stock', 'estado', 'activo']
    readonly_fields = ['fecha_creacion', 'fecha_actualizacion', 'margen_ganancia_display']
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('nombre', 'descripcion', 'codigo', 'categoria', 'marca')
        }),
        ('Precios', {
            'fields': ('precio_compra', 'precio_venta', 'moneda', 'margen_ganancia_display')
        }),
        ('Inventario', {
            'fields': ('stock', 'stock_minimo', 'estado')
        }),
        ('Imágenes', {
            'fields': ('imagen_principal', 'imagen_2', 'imagen_3')
        }),
        ('Características', {
            'fields': ('peso', 'dimensiones'),
            'classes': ('collapse',)
        }),
        ('Configuración', {
            'fields': ('activo', 'destacado', 'creado_por')
        }),
        ('Fechas', {
            'fields': ('fecha_creacion', 'fecha_actualizacion'),
            'classes': ('collapse',)
        })
    )
    
    inlines = [ImagenProductoInline]
    
    def precio_venta_formateado(self, obj):
        return f"{obj.precio_venta} {obj.moneda}"
    precio_venta_formateado.short_description = 'Precio'
    precio_venta_formateado.admin_order_field = 'precio_venta'
    
    def stock_status(self, obj):
        if obj.stock == 0:
            color = 'red'
            status = 'Sin stock'
        elif obj.stock_bajo:
            color = 'orange'
            status = 'Stock bajo'
        else:
            color = 'green'
            status = 'Stock OK'
        
        return format_html(
            '<span style="color: {};">{}</span>',
            color, status
        )
    stock_status.short_description = 'Estado Stock'
    
    def margen_ganancia_display(self, obj):
        margen = obj.margen_ganancia
        color = 'green' if margen > 20 else 'orange' if margen > 0 else 'red'
        return format_html(
            '<span style="color: {};">{:.1f}%</span>',
            color, margen
        )
    margen_ganancia_display.short_description = 'Margen de Ganancia'
    
    def save_model(self, request, obj, form, change):
        if not change:  # Si es un nuevo objeto
            obj.creado_por = request.user
        super().save_model(request, obj, form, change)
    
    actions = ['marcar_como_destacado', 'marcar_como_no_destacado', 'activar_productos', 'desactivar_productos']
    
    def marcar_como_destacado(self, request, queryset):
        queryset.update(destacado=True)
        self.message_user(request, f"{queryset.count()} productos marcados como destacados.")
    marcar_como_destacado.short_description = "Marcar como destacado"
    
    def marcar_como_no_destacado(self, request, queryset):
        queryset.update(destacado=False)
        self.message_user(request, f"{queryset.count()} productos desmarcados como destacados.")
    marcar_como_no_destacado.short_description = "Desmarcar como destacado"
    
    def activar_productos(self, request, queryset):
        queryset.update(activo=True)
        self.message_user(request, f"{queryset.count()} productos activados.")
    activar_productos.short_description = "Activar productos"
    
    def desactivar_productos(self, request, queryset):
        queryset.update(activo=False)
        self.message_user(request, f"{queryset.count()} productos desactivados.")
    desactivar_productos.short_description = "Desactivar productos"

@admin.register(ImagenProducto)
class ImagenProductoAdmin(admin.ModelAdmin):
    list_display = ['producto', 'descripcion', 'orden']
    list_filter = ['producto__categoria']
    search_fields = ['producto__nombre', 'descripcion']