from django.urls import path
from . import views

app_name = 'usuarios'

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('perfil/', views.perfil_usuario, name='perfil'),
    
    # URLs para administradores
    path('lista/', views.lista_usuarios, name='lista_usuarios'),
    path('crear/', views.crear_usuario, name='crear_usuario'),
    path('toggle-activo/<int:usuario_id>/', views.toggle_usuario_activo, name='toggle_usuario_activo'),
    path('sesiones-activas/', views.sesiones_activas, name='sesiones_activas'),
]