from django.urls import path
from . import views

app_name = 'productos'

urlpatterns = [
    path('', views.home, name='home'),
    path('catalogo/', views.catalogo, name='catalogo'),
    path('producto/<int:pk>/', views.producto_detalle, name='detalle'),
    path('categoria/<int:pk>/', views.categoria_detalle, name='categoria'),
]