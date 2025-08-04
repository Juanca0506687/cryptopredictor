from django.urls import path
from . import views

app_name = 'ventas'

urlpatterns = [
    # Ventas
    path('', views.VentaListView.as_view(), name='lista'),
    path('crear/', views.VentaCreateView.as_view(), name='crear'),
    path('<int:pk>/', views.VentaDetailView.as_view(), name='detalle'),
    path('<int:pk>/editar/', views.VentaUpdateView.as_view(), name='editar'),
    path('<int:pk>/factura/', views.generar_factura_pdf, name='factura_pdf'),
    
    # Pagos
    path('<int:venta_pk>/pagos/', views.PagoListView.as_view(), name='pagos_lista'),
    path('<int:venta_pk>/pagos/crear/', views.PagoCreateView.as_view(), name='pagos_crear'),
    
    # Estados
    path('<int:pk>/cambiar-estado/', views.cambiar_estado_venta, name='cambiar_estado'),
    
    # API
    path('api/calcular-total/', views.calcular_total_venta, name='api_calcular_total'),
]