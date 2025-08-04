from django.urls import path
from . import views

app_name = 'productos'

urlpatterns = [
    # Monturas
    path('monturas/', views.MonturaListView.as_view(), name='monturas_lista'),
    path('monturas/crear/', views.MonturaCreateView.as_view(), name='monturas_crear'),
    path('monturas/<int:pk>/', views.MonturaDetailView.as_view(), name='monturas_detalle'),
    path('monturas/<int:pk>/editar/', views.MonturaUpdateView.as_view(), name='monturas_editar'),
    
    # Cristales
    path('cristales/', views.CristalListView.as_view(), name='cristales_lista'),
    path('cristales/crear/', views.CristalCreateView.as_view(), name='cristales_crear'),
    path('cristales/<int:pk>/', views.CristalDetailView.as_view(), name='cristales_detalle'),
    path('cristales/<int:pk>/editar/', views.CristalUpdateView.as_view(), name='cristales_editar'),
    
    # Inventario
    path('inventario/', views.InventarioView.as_view(), name='inventario'),
    path('movimientos/', views.MovimientoInventarioListView.as_view(), name='movimientos_lista'),
    path('movimientos/crear/', views.MovimientoInventarioCreateView.as_view(), name='movimientos_crear'),
    
    # Categorías y Proveedores
    path('categorias/', views.CategoriaListView.as_view(), name='categorias_lista'),
    path('proveedores/', views.ProveedorListView.as_view(), name='proveedores_lista'),
    
    # API
    path('api/buscar-productos/', views.buscar_productos, name='api_buscar_productos'),
]