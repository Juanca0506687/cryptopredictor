from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import PerfilUsuario

class PerfilUsuarioInline(admin.StackedInline):
    model = PerfilUsuario
    can_delete = False
    verbose_name_plural = 'Perfil'

class UserAdmin(BaseUserAdmin):
    inlines = (PerfilUsuarioInline,)

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)

@admin.register(PerfilUsuario)
class PerfilUsuarioAdmin(admin.ModelAdmin):
    list_display = ['user', 'tipo_usuario', 'telefono', 'ciudad', 'activo', 'fecha_creacion']
    list_filter = ['tipo_usuario', 'activo', 'ciudad', 'provincia']
    search_fields = ['user__username', 'user__first_name', 'user__last_name', 'telefono']
    readonly_fields = ['fecha_creacion', 'fecha_actualizacion']