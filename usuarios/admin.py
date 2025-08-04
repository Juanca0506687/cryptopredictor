from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _
from .models import Usuario, PerfilUsuario, SesionUsuario


class PerfilUsuarioInline(admin.StackedInline):
    model = PerfilUsuario
    can_delete = False
    verbose_name_plural = 'Perfil'


@admin.register(Usuario)
class UsuarioAdmin(BaseUserAdmin):
    inlines = (PerfilUsuarioInline,)
    
    list_display = ('username', 'email', 'first_name', 'last_name', 'rol', 'activo', 'is_staff')
    list_filter = ('rol', 'activo', 'is_staff', 'is_superuser', 'date_joined')
    search_fields = ('username', 'first_name', 'last_name', 'email', 'cedula')
    ordering = ('username',)
    
    fieldsets = BaseUserAdmin.fieldsets + (
        (_('Información Adicional'), {
            'fields': ('rol', 'telefono', 'cedula', 'direccion', 'activo')
        }),
    )
    
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        (_('Información Adicional'), {
            'fields': ('rol', 'telefono', 'cedula', 'direccion', 'activo')
        }),
    )


@admin.register(PerfilUsuario)
class PerfilUsuarioAdmin(admin.ModelAdmin):
    list_display = ('usuario', 'fecha_nacimiento', 'salario', 'comision_ventas')
    list_filter = ('usuario__rol',)
    search_fields = ('usuario__username', 'usuario__first_name', 'usuario__last_name')


@admin.register(SesionUsuario)
class SesionUsuarioAdmin(admin.ModelAdmin):
    list_display = ('usuario', 'fecha_inicio', 'fecha_fin', 'ip_address', 'activa')
    list_filter = ('activa', 'fecha_inicio', 'usuario__rol')
    search_fields = ('usuario__username', 'ip_address')
    readonly_fields = ('fecha_inicio', 'ip_address', 'user_agent')
    
    def has_add_permission(self, request):
        return False
