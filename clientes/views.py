from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.http import JsonResponse
from .models import Cliente, RecetaMedica

class ClienteListView(LoginRequiredMixin, ListView):
    model = Cliente
    template_name = 'clientes/lista.html'
    context_object_name = 'clientes'
    paginate_by = 20

class ClienteDetailView(LoginRequiredMixin, DetailView):
    model = Cliente
    template_name = 'clientes/detalle.html'

class ClienteCreateView(LoginRequiredMixin, CreateView):
    model = Cliente
    template_name = 'clientes/crear.html'
    fields = '__all__'

class ClienteUpdateView(LoginRequiredMixin, UpdateView):
    model = Cliente
    template_name = 'clientes/editar.html'
    fields = '__all__'

class RecetaListView(LoginRequiredMixin, ListView):
    model = RecetaMedica
    template_name = 'clientes/recetas_lista.html'
    context_object_name = 'recetas'

class RecetaDetailView(LoginRequiredMixin, DetailView):
    model = RecetaMedica
    template_name = 'clientes/recetas_detalle.html'

class RecetaCreateView(LoginRequiredMixin, CreateView):
    model = RecetaMedica
    template_name = 'clientes/recetas_crear.html'
    fields = '__all__'

class RecetaUpdateView(LoginRequiredMixin, UpdateView):
    model = RecetaMedica
    template_name = 'clientes/recetas_editar.html'
    fields = '__all__'

def buscar_clientes(request):
    query = request.GET.get('q', '')
    clientes = []
    
    if query:
        resultados = Cliente.objects.filter(
            nombres__icontains=query
        ) | Cliente.objects.filter(
            apellidos__icontains=query
        ) | Cliente.objects.filter(
            numero_documento__icontains=query
        )
        
        for cliente in resultados[:10]:
            clientes.append({
                'id': cliente.id,
                'nombre': cliente.nombre_completo,
                'documento': cliente.numero_documento,
                'telefono': cliente.telefono or ''
            })
    
    return JsonResponse({'clientes': clientes})
