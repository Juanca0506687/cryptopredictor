from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'usuarios'

urlpatterns = [
    # Autenticación
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('cambiar-password/', auth_views.PasswordChangeView.as_view(
        template_name='usuarios/cambiar_password.html',
        success_url='/dashboard/'
    ), name='cambiar_password'),
    
    # Gestión de usuarios (solo admin)
    path('', views.UsuarioListView.as_view(), name='lista'),
    path('crear/', views.UsuarioCreateView.as_view(), name='crear'),
    path('<int:pk>/', views.UsuarioDetailView.as_view(), name='detalle'),
    path('<int:pk>/editar/', views.UsuarioUpdateView.as_view(), name='editar'),
    path('<int:pk>/toggle/', views.toggle_usuario_activo, name='toggle_activo'),
    
    # Perfil
    path('perfil/', views.PerfilView.as_view(), name='perfil'),
    path('perfil/editar/', views.EditarPerfilView.as_view(), name='editar_perfil'),
]