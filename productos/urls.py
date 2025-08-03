from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('catalogo/', views.catalogo_view, name='catalogo'),
    path('producto/<int:pk>/', views.producto_detalle_view, name='producto_detalle'),
    path('categoria/<int:pk>/', views.categoria_view, name='categoria'),
]