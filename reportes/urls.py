from django.urls import path
from . import views

app_name = 'reportes'

urlpatterns = [
    # Dashboard
    path('', views.DashboardView.as_view(), name='dashboard'),
    
    # Reportes
    path('lista/', views.ReporteListView.as_view(), name='lista'),
    path('generar/', views.GenerarReporteView.as_view(), name='generar'),
    path('<int:pk>/descargar/', views.descargar_reporte, name='descargar'),
    path('<int:pk>/eliminar/', views.eliminar_reporte, name='eliminar'),
    
    # Estadísticas
    path('estadisticas/ventas/', views.estadisticas_ventas, name='estadisticas_ventas'),
    path('estadisticas/inventario/', views.estadisticas_inventario, name='estadisticas_inventario'),
    path('estadisticas/taller/', views.estadisticas_taller, name='estadisticas_taller'),
    
    # API para gráficos
    path('api/ventas-mes/', views.ventas_por_mes, name='api_ventas_mes'),
    path('api/productos-populares/', views.productos_populares, name='api_productos_populares'),
    path('api/estado-ordenes/', views.estado_ordenes_taller, name='api_estado_ordenes'),
]