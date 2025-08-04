from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.http import JsonResponse
from .models import OrdenTrabajo, TareaTaller, ControlCalidad, RecursoTaller

class OrdenTrabajoListView(LoginRequiredMixin, ListView):
    model = OrdenTrabajo
    template_name = 'taller/ordenes_lista.html'
    context_object_name = 'ordenes'
    paginate_by = 20

class OrdenTrabajoDetailView(LoginRequiredMixin, DetailView):
    model = OrdenTrabajo
    template_name = 'taller/ordenes_detalle.html'

class OrdenTrabajoUpdateView(LoginRequiredMixin, UpdateView):
    model = OrdenTrabajo
    template_name = 'taller/ordenes_editar.html'
    fields = ['estado', 'prioridad', 'tecnico_asignado', 'observaciones_tecnicas']

class TareaListView(LoginRequiredMixin, ListView):
    model = TareaTaller
    template_name = 'taller/tareas_lista.html'
    context_object_name = 'tareas'

class ControlCalidadCreateView(LoginRequiredMixin, CreateView):
    model = ControlCalidad
    template_name = 'taller/control_calidad_crear.html'
    fields = '__all__'

class RecursoTallerListView(LoginRequiredMixin, ListView):
    model = RecursoTaller
    template_name = 'taller/recursos_lista.html'
    context_object_name = 'recursos'

class RecursoTallerCreateView(LoginRequiredMixin, CreateView):
    model = RecursoTaller
    template_name = 'taller/recursos_crear.html'
    fields = '__all__'

@login_required
def completar_tarea(request, pk):
    if request.method == 'POST':
        tarea = get_object_or_404(TareaTaller, pk=pk)
        observaciones = request.POST.get('observaciones', '')
        tarea.marcar_completada(request.user, observaciones)
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@login_required
def asignar_tecnico(request):
    if request.method == 'POST':
        orden_id = request.POST.get('orden_id')
        tecnico_id = request.POST.get('tecnico_id')
        
        orden = get_object_or_404(OrdenTrabajo, pk=orden_id)
        from usuarios.models import Usuario
        tecnico = get_object_or_404(Usuario, pk=tecnico_id)
        
        orden.tecnico_asignado = tecnico
        orden.save()
        
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})
