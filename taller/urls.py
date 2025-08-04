from django.urls import path
from . import views

app_name = 'taller'

urlpatterns = [
    # Órdenes de trabajo
    path('ordenes/', views.OrdenTrabajoListView.as_view(), name='ordenes_lista'),
    path('ordenes/<int:pk>/', views.OrdenTrabajoDetailView.as_view(), name='ordenes_detalle'),
    path('ordenes/<int:pk>/editar/', views.OrdenTrabajoUpdateView.as_view(), name='ordenes_editar'),
    
    # Tareas
    path('ordenes/<int:orden_pk>/tareas/', views.TareaListView.as_view(), name='tareas_lista'),
    path('tareas/<int:pk>/completar/', views.completar_tarea, name='tareas_completar'),
    
    # Control de calidad
    path('ordenes/<int:orden_pk>/control-calidad/', views.ControlCalidadCreateView.as_view(), name='control_calidad_crear'),
    
    # Recursos
    path('recursos/', views.RecursoTallerListView.as_view(), name='recursos_lista'),
    path('recursos/crear/', views.RecursoTallerCreateView.as_view(), name='recursos_crear'),
    
    # API
    path('api/asignar-tecnico/', views.asignar_tecnico, name='api_asignar_tecnico'),
]