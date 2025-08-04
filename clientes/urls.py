from django.urls import path
from . import views

app_name = 'clientes'

urlpatterns = [
    # Clientes
    path('', views.ClienteListView.as_view(), name='lista'),
    path('crear/', views.ClienteCreateView.as_view(), name='crear'),
    path('<int:pk>/', views.ClienteDetailView.as_view(), name='detalle'),
    path('<int:pk>/editar/', views.ClienteUpdateView.as_view(), name='editar'),
    
    # Recetas
    path('<int:cliente_pk>/recetas/', views.RecetaListView.as_view(), name='recetas_lista'),
    path('<int:cliente_pk>/recetas/crear/', views.RecetaCreateView.as_view(), name='recetas_crear'),
    path('recetas/<int:pk>/', views.RecetaDetailView.as_view(), name='recetas_detalle'),
    path('recetas/<int:pk>/editar/', views.RecetaUpdateView.as_view(), name='recetas_editar'),
    
    # API
    path('api/buscar/', views.buscar_clientes, name='api_buscar'),
]