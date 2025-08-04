from django.contrib import admin
from .models import (
    Categoria, Proveedor, Montura, TipoCristal, Cristal, MovimientoInventario
)

@admin.register(Categoria)
class CategoriaAdmin(admin.ModelAdmin):
    list_display = ('nombre', 'activo')
    list_filter = ('activo',)
    search_fields = ('nombre',)

@admin.register(Proveedor)
class ProveedorAdmin(admin.ModelAdmin):
    list_display = ('nombre', 'contacto', 'telefono', 'email', 'activo')
    list_filter = ('activo',)
    search_fields = ('nombre', 'contacto', 'email')

@admin.register(Montura)
class MonturaAdmin(admin.ModelAdmin):
    list_display = ('codigo', 'nombre', 'categoria', 'material', 'precio_venta', 'stock_actual', 'necesita_restock')
    list_filter = ('categoria', 'material', 'genero', 'proveedor', 'activo')
    search_fields = ('codigo', 'nombre', 'color')
    readonly_fields = ('fecha_creacion', 'fecha_actualizacion')
    
    def necesita_restock(self, obj):
        return obj.necesita_restock
    necesita_restock.boolean = True
    necesita_restock.short_description = 'Necesita Restock'

@admin.register(TipoCristal)
class TipoCristalAdmin(admin.ModelAdmin):
    list_display = ('nombre', 'precio_base', 'activo')
    list_filter = ('activo',)
    search_fields = ('nombre',)

@admin.register(Cristal)
class CristalAdmin(admin.ModelAdmin):
    list_display = ('codigo', 'tipo_cristal', 'tipo_graduacion', 'precio_venta', 'stock_actual', 'necesita_restock')
    list_filter = ('tipo_cristal', 'tipo_graduacion', 'proveedor', 'activo')
    search_fields = ('codigo',)
    readonly_fields = ('fecha_creacion', 'fecha_actualizacion')
    
    def necesita_restock(self, obj):
        return obj.necesita_restock
    necesita_restock.boolean = True
    necesita_restock.short_description = 'Necesita Restock'

@admin.register(MovimientoInventario)
class MovimientoInventarioAdmin(admin.ModelAdmin):
    list_display = ('fecha', 'tipo_movimiento', 'get_producto', 'cantidad', 'usuario')
    list_filter = ('tipo_movimiento', 'fecha', 'usuario')
    readonly_fields = ('fecha',)
    
    def get_producto(self, obj):
        return obj.montura or obj.cristal
    get_producto.short_description = 'Producto'
