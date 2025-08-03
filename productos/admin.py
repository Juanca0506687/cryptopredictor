from django.contrib import admin
from django.utils.html import format_html
from .models import Categoria, Producto, ImagenProducto

@admin.register(Categoria)
class CategoriaAdmin(admin.ModelAdmin):
    list_display = ['nombre', 'activa', 'fecha_creacion', 'productos_count']
    list_filter = ['activa', 'fecha_creacion']
    search_fields = ['nombre', 'descripcion']
    prepopulated_fields = {'nombre': ('nombre',)}
    
    def productos_count(self, obj):
        return obj.producto_set.count()
    productos_count.short_description = 'Productos'

class ImagenProductoInline(admin.TabularInline):
    model = ImagenProducto
    extra = 1
    fields = ['imagen', 'descripcion', 'orden']

@admin.register(Producto)
class ProductoAdmin(admin.ModelAdmin):
    list_display = [
        'codigo', 'nombre', 'categoria', 'precio_venta_formatted', 
        'moneda', 'stock', 'estado', 'destacado', 'margen_display'
    ]
    list_filter = ['categoria', 'moneda', 'estado', 'destacado', 'fecha_creacion']
    search_fields = ['nombre', 'codigo', 'descripcion']
    list_editable = ['stock', 'estado', 'destacado']
    readonly_fields = ['fecha_creacion', 'fecha_modificacion', 'vistas', 'margen_display', 'imagen_preview']
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('nombre', 'descripcion', 'codigo', 'categoria')
        }),
        ('Precios y Moneda', {
            'fields': ('precio_compra', 'precio_venta', 'moneda', 'margen_display'),
            'classes': ('wide',)
        }),
        ('Inventario', {
            'fields': ('stock', 'stock_minimo', 'estado'),
            'classes': ('wide',)
        }),
        ('Imágenes', {
            'fields': ('imagen_principal', 'imagen_preview', 'imagen_2', 'imagen_3'),
            'classes': ('wide',)
        }),
        ('Configuración', {
            'fields': ('destacado',),
        }),
        ('Estadísticas', {
            'fields': ('vistas', 'ventas_totales', 'fecha_creacion', 'fecha_modificacion'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [ImagenProductoInline]
    
    actions = ['marcar_destacado', 'desmarcar_destacado', 'activar_productos', 'desactivar_productos']
    
    def precio_venta_formatted(self, obj):
        return f"{obj.precio_venta} {obj.moneda}"
    precio_venta_formatted.short_description = 'Precio Venta'
    precio_venta_formatted.admin_order_field = 'precio_venta'
    
    def margen_display(self, obj):
        margen = obj.margen_ganancia
        color = 'green' if margen > 20 else 'orange' if margen > 0 else 'red'
        return format_html(
            '<span style="color: {};">{:.2f}%</span>',
            color, margen
        )
    margen_display.short_description = 'Margen'
    
    def imagen_preview(self, obj):
        if obj.imagen_principal:
            return format_html(
                '<img src="{}" width="100" height="100" style="object-fit: cover;" />',
                obj.imagen_principal.url
            )
        return "Sin imagen"
    imagen_preview.short_description = 'Vista previa'
    
    def marcar_destacado(self, request, queryset):
        updated = queryset.update(destacado=True)
        self.message_user(request, f'{updated} productos marcados como destacados.')
    marcar_destacado.short_description = "Marcar como destacado"
    
    def desmarcar_destacado(self, request, queryset):
        updated = queryset.update(destacado=False)
        self.message_user(request, f'{updated} productos desmarcados como destacados.')
    desmarcar_destacado.short_description = "Desmarcar como destacado"
    
    def activar_productos(self, request, queryset):
        updated = queryset.update(estado='activo')
        self.message_user(request, f'{updated} productos activados.')
    activar_productos.short_description = "Activar productos"
    
    def desactivar_productos(self, request, queryset):
        updated = queryset.update(estado='inactivo')
        self.message_user(request, f'{updated} productos desactivados.')
    desactivar_productos.short_description = "Desactivar productos"

# Personalización del admin
admin.site.site_header = "Tienda Cuba - Administración"
admin.site.site_title = "Tienda Cuba Admin"
admin.site.index_title = "Panel de Administración"
