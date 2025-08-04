from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, TemplateView
from django.http import JsonResponse
from .models import Montura, Cristal, Categoria, Proveedor, MovimientoInventario

class MonturaListView(LoginRequiredMixin, ListView):
    model = Montura
    template_name = 'productos/monturas_lista.html'
    context_object_name = 'monturas'
    paginate_by = 20

class MonturaDetailView(LoginRequiredMixin, DetailView):
    model = Montura
    template_name = 'productos/monturas_detalle.html'

class MonturaCreateView(LoginRequiredMixin, CreateView):
    model = Montura
    template_name = 'productos/monturas_crear.html'
    fields = '__all__'

class MonturaUpdateView(LoginRequiredMixin, UpdateView):
    model = Montura
    template_name = 'productos/monturas_editar.html'
    fields = '__all__'

class CristalListView(LoginRequiredMixin, ListView):
    model = Cristal
    template_name = 'productos/cristales_lista.html'
    context_object_name = 'cristales'
    paginate_by = 20

class CristalDetailView(LoginRequiredMixin, DetailView):
    model = Cristal
    template_name = 'productos/cristales_detalle.html'

class CristalCreateView(LoginRequiredMixin, CreateView):
    model = Cristal
    template_name = 'productos/cristales_crear.html'
    fields = '__all__'

class CristalUpdateView(LoginRequiredMixin, UpdateView):
    model = Cristal
    template_name = 'productos/cristales_editar.html'
    fields = '__all__'

class InventarioView(LoginRequiredMixin, TemplateView):
    template_name = 'productos/inventario.html'

class MovimientoInventarioListView(LoginRequiredMixin, ListView):
    model = MovimientoInventario
    template_name = 'productos/movimientos_lista.html'
    context_object_name = 'movimientos'
    paginate_by = 20

class MovimientoInventarioCreateView(LoginRequiredMixin, CreateView):
    model = MovimientoInventario
    template_name = 'productos/movimientos_crear.html'
    fields = '__all__'

class CategoriaListView(LoginRequiredMixin, ListView):
    model = Categoria
    template_name = 'productos/categorias_lista.html'
    context_object_name = 'categorias'

class ProveedorListView(LoginRequiredMixin, ListView):
    model = Proveedor
    template_name = 'productos/proveedores_lista.html'
    context_object_name = 'proveedores'

def buscar_productos(request):
    query = request.GET.get('q', '')
    productos = []
    
    if query:
        monturas = Montura.objects.filter(nombre__icontains=query)[:10]
        cristales = Cristal.objects.filter(codigo__icontains=query)[:10]
        
        for montura in monturas:
            productos.append({
                'id': montura.id,
                'tipo': 'montura',
                'nombre': montura.nombre,
                'codigo': montura.codigo,
                'precio': float(montura.precio_venta),
                'stock': montura.stock_actual
            })
        
        for cristal in cristales:
            productos.append({
                'id': cristal.id,
                'tipo': 'cristal',
                'nombre': str(cristal),
                'codigo': cristal.codigo,
                'precio': float(cristal.precio_venta),
                'stock': cristal.stock_actual
            })
    
    return JsonResponse({'productos': productos})
