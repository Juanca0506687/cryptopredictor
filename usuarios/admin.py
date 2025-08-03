from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import PerfilUsuario

class PerfilUsuarioInline(admin.StackedInline):
    model = PerfilUsuario
    can_delete = False
    verbose_name_plural = 'Perfil'

class CustomUserAdmin(UserAdmin):
    inlines = (PerfilUsuarioInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'get_tipo_usuario')
    list_filter = UserAdmin.list_filter + ('perfil__tipo_usuario',)
    
    def get_tipo_usuario(self, obj):
        return obj.perfil.tipo_usuario if hasattr(obj, 'perfil') else 'Sin perfil'
    get_tipo_usuario.short_description = 'Tipo Usuario'

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

@admin.register(PerfilUsuario)
class PerfilUsuarioAdmin(admin.ModelAdmin):
    list_display = ['user', 'nombre_completo', 'tipo_usuario', 'telefono', 'activo', 'fecha_registro']
    list_filter = ['tipo_usuario', 'activo', 'fecha_registro']
    search_fields = ['user__username', 'user__first_name', 'user__last_name', 'telefono']
    readonly_fields = ['fecha_registro']
