from django.contrib import admin
from django.utils.html import format_html
from .models import (
    CategoriaProducto, Marca, Producto, Montura, Cristal,
    MovimientoInventario, AlertaStock
)


@admin.register(CategoriaProducto)
class CategoriaProductoAdmin(admin.ModelAdmin):
    list_display = ('nombre', 'activa', 'productos_count')
    list_filter = ('activa',)
    search_fields = ('nombre', 'descripcion')
    
    def productos_count(self, obj):
        return obj.producto_set.count()
    productos_count.short_description = 'Productos'


@admin.register(Marca)
class MarcaAdmin(admin.ModelAdmin):
    list_display = ('nombre', 'pais_origen', 'activa', 'productos_count')
    list_filter = ('activa', 'pais_origen')
    search_fields = ('nombre', 'pais_origen')
    
    def productos_count(self, obj):
        return obj.producto_set.count()
    productos_count.short_description = 'Productos'


class MonturaInline(admin.StackedInline):
    model = Montura
    extra = 0


class CristalInline(admin.StackedInline):
    model = Cristal
    extra = 0


class MovimientoInventarioInline(admin.TabularInline):
    model = MovimientoInventario
    extra = 0
    readonly_fields = ('fecha', 'usuario')
    fields = ('tipo', 'cantidad', 'motivo', 'fecha', 'usuario', 'activo')


@admin.register(Producto)
class ProductoAdmin(admin.ModelAdmin):
    list_display = (
        'codigo', 'nombre', 'tipo', 'categoria', 'marca',
        'precio_venta_cup', 'stock_actual_display', 'necesita_reposicion_display',
        'activo'
    )
    list_filter = ('tipo', 'categoria', 'marca', 'activo')
    search_fields = ('codigo', 'nombre', 'descripcion')
    readonly_fields = ('fecha_creacion', 'fecha_actualizacion', 'creado_por')
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('codigo', 'nombre', 'descripcion', 'tipo', 'categoria', 'marca')
        }),
        ('Precios', {
            'fields': (
                'precio_costo', 'precio_venta_cup', 'precio_venta_usd',
                'precio_venta_eur', 'precio_venta_mlc'
            )
        }),
        ('Inventario', {
            'fields': ('stock_minimo', 'activo')
        }),
        ('Metadatos', {
            'fields': ('fecha_creacion', 'fecha_actualizacion', 'creado_por'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [MonturaInline, CristalInline, MovimientoInventarioInline]
    
    def save_model(self, request, obj, form, change):
        if not change:  # Si es un nuevo objeto
            obj.creado_por = request.user
        super().save_model(request, obj, form, change)
    
    def stock_actual_display(self, obj):
        stock = obj.stock_actual
        if stock <= obj.stock_minimo:
            return format_html(
                '<span style="color: red; font-weight: bold;">{}</span>',
                stock
            )
        return stock
    stock_actual_display.short_description = 'Stock Actual'
    
    def necesita_reposicion_display(self, obj):
        if obj.necesita_reposicion:
            return format_html(
                '<span style="color: red;">⚠️ Sí</span>'
            )
        return format_html(
            '<span style="color: green;">✓ No</span>'
        )
    necesita_reposicion_display.short_description = 'Necesita Reposición'


@admin.register(Montura)
class MonturaAdmin(admin.ModelAdmin):
    list_display = (
        'producto', 'material', 'color', 'genero',
        'ancho_lente', 'puente', 'varilla'
    )
    list_filter = ('material', 'genero', 'color')
    search_fields = ('producto__nombre', 'producto__codigo', 'color')


@admin.register(Cristal)
class CristalAdmin(admin.ModelAdmin):
    list_display = (
        'producto', 'tipo', 'material', 'indice_refraccion',
        'anti_reflejo', 'fotocromatico', 'polarizado', 'filtro_azul'
    )
    list_filter = (
        'tipo', 'material', 'anti_reflejo', 'fotocromatico',
        'polarizado', 'filtro_azul'
    )
    search_fields = ('producto__nombre', 'producto__codigo')


@admin.register(MovimientoInventario)
class MovimientoInventarioAdmin(admin.ModelAdmin):
    list_display = (
        'producto', 'tipo', 'cantidad', 'motivo',
        'fecha', 'usuario', 'activo'
    )
    list_filter = ('tipo', 'fecha', 'usuario', 'activo')
    search_fields = ('producto__nombre', 'producto__codigo', 'motivo')
    readonly_fields = ('fecha',)
    
    def save_model(self, request, obj, form, change):
        if not change:  # Si es un nuevo objeto
            obj.usuario = request.user
        super().save_model(request, obj, form, change)


@admin.register(AlertaStock)
class AlertaStockAdmin(admin.ModelAdmin):
    list_display = (
        'producto', 'mensaje', 'fecha_creacion',
        'vista', 'resuelta'
    )
    list_filter = ('vista', 'resuelta', 'fecha_creacion')
    search_fields = ('producto__nombre', 'producto__codigo', 'mensaje')
    readonly_fields = ('fecha_creacion',)
    
    actions = ['marcar_como_vista', 'marcar_como_resuelta']
    
    def marcar_como_vista(self, request, queryset):
        queryset.update(vista=True)
        self.message_user(request, f'{queryset.count()} alertas marcadas como vistas.')
    marcar_como_vista.short_description = 'Marcar como vista'
    
    def marcar_como_resuelta(self, request, queryset):
        queryset.update(resuelta=True)
        self.message_user(request, f'{queryset.count()} alertas marcadas como resueltas.')
    marcar_como_resuelta.short_description = 'Marcar como resuelta'
