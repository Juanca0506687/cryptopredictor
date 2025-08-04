from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.http import JsonResponse, HttpResponse
from .models import Venta, Pago

class VentaListView(LoginRequiredMixin, ListView):
    model = Venta
    template_name = 'ventas/lista.html'
    context_object_name = 'ventas'
    paginate_by = 20

class VentaDetailView(LoginRequiredMixin, DetailView):
    model = Venta
    template_name = 'ventas/detalle.html'

class VentaCreateView(LoginRequiredMixin, CreateView):
    model = Venta
    template_name = 'ventas/crear.html'
    fields = ['cliente', 'receta', 'observaciones']
    
    def form_valid(self, form):
        form.instance.vendedor = self.request.user
        return super().form_valid(form)

class VentaUpdateView(LoginRequiredMixin, UpdateView):
    model = Venta
    template_name = 'ventas/editar.html'
    fields = ['cliente', 'receta', 'observaciones']

class PagoListView(LoginRequiredMixin, ListView):
    model = Pago
    template_name = 'ventas/pagos_lista.html'
    context_object_name = 'pagos'

class PagoCreateView(LoginRequiredMixin, CreateView):
    model = Pago
    template_name = 'ventas/pagos_crear.html'
    fields = '__all__'

@login_required
def generar_factura_pdf(request, pk):
    venta = get_object_or_404(Venta, pk=pk)
    
    # Aquí iría la lógica para generar el PDF
    # Por ahora, devolvemos una respuesta simple
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="factura_{venta.numero_factura}.pdf"'
    response.write(b'PDF content would go here')
    return response

@login_required
def cambiar_estado_venta(request, pk):
    if request.method == 'POST':
        venta = get_object_or_404(Venta, pk=pk)
        nuevo_estado = request.POST.get('estado')
        if nuevo_estado in dict(Venta.ESTADOS):
            venta.estado = nuevo_estado
            venta.save()
            return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@login_required
def calcular_total_venta(request):
    # Lógica para calcular el total de una venta
    return JsonResponse({'total': 0})
