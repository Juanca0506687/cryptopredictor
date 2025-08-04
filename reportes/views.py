from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic import TemplateView, ListView, CreateView
from django.http import JsonResponse, HttpResponse, Http404
from django.db.models import Sum, Count, Q, Avg
from django.utils import timezone
from datetime import date, timedelta, datetime
from decimal import Decimal
import json

from .models import Reporte, TipoReporte, EstadisticaVenta, EstadisticaInventario, EstadisticaTaller
from productos.models import Montura, Cristal
from ventas.models import Venta, ItemVenta
from clientes.models import Cliente
from taller.models import OrdenTrabajo
from usuarios.models import Usuario

class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'reportes/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        hoy = date.today()
        hace_30_dias = hoy - timedelta(days=30)
        
        # Estadísticas generales
        context['total_clientes'] = Cliente.objects.filter(activo=True).count()
        context['total_productos'] = (
            Montura.objects.filter(activo=True).count() + 
            Cristal.objects.filter(activo=True).count()
        )
        
        # Ventas del mes
        ventas_mes = Venta.objects.filter(
            fecha__gte=hace_30_dias,
            estado__in=['confirmada', 'entregada']
        )
        context['ventas_mes'] = ventas_mes.count()
        context['ingresos_mes'] = ventas_mes.aggregate(
            total=Sum('total')
        )['total'] or Decimal('0')
        
        # Órdenes de taller
        ordenes = OrdenTrabajo.objects.all()
        context['ordenes_pendientes'] = ordenes.filter(estado='pendiente').count()
        context['ordenes_proceso'] = ordenes.filter(estado='en_proceso').count()
        context['ordenes_atrasadas'] = ordenes.filter(
            fecha_fin_estimada__lt=timezone.now(),
            estado__in=['pendiente', 'en_proceso']
        ).count()
        
        # Productos con bajo stock
        from django.db import models
        monturas_bajo_stock = Montura.objects.filter(
            stock_actual__lte=models.F('stock_minimo'),
            activo=True
        )
        cristales_bajo_stock = Cristal.objects.filter(
            stock_actual__lte=models.F('stock_minimo'),
            activo=True
        )
        context['productos_bajo_stock'] = (
            monturas_bajo_stock.count() + cristales_bajo_stock.count()
        )
        
        # Ventas recientes
        context['ventas_recientes'] = Venta.objects.select_related(
            'cliente', 'vendedor'
        ).order_by('-fecha')[:5]
        
        # Clientes recientes
        context['clientes_recientes'] = Cliente.objects.filter(
            activo=True
        ).order_by('-fecha_registro')[:5]
        
        # Top vendedores del mes
        top_vendedores = Venta.objects.filter(
            fecha__gte=hace_30_dias,
            estado__in=['confirmada', 'entregada']
        ).values('vendedor__username', 'vendedor__first_name', 'vendedor__last_name').annotate(
            total_ventas=Sum('total'),
            cantidad_ventas=Count('id')
        ).order_by('-total_ventas')[:5]
        context['top_vendedores'] = top_vendedores
        
        return context

class ReporteListView(LoginRequiredMixin, ListView):
    model = Reporte
    template_name = 'reportes/lista.html'
    context_object_name = 'reportes'
    paginate_by = 20
    
    def get_queryset(self):
        return Reporte.objects.filter(
            usuario_solicita=self.request.user
        ).order_by('-fecha_solicitud')

class GenerarReporteView(LoginRequiredMixin, CreateView):
    model = Reporte
    template_name = 'reportes/generar.html'
    fields = ['tipo_reporte', 'fecha_inicio', 'fecha_fin', 'formato', 'titulo']
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['tipos_reporte'] = TipoReporte.objects.filter(activo=True)
        return context
    
    def form_valid(self, form):
        form.instance.usuario_solicita = self.request.user
        return super().form_valid(form)

@login_required
def descargar_reporte(request, pk):
    reporte = get_object_or_404(Reporte, pk=pk, usuario_solicita=request.user)
    
    if not reporte.esta_listo:
        raise Http404("El reporte no está disponible")
    
    response = HttpResponse(
        reporte.archivo.read(),
        content_type='application/octet-stream'
    )
    response['Content-Disposition'] = f'attachment; filename="{reporte.archivo.name}"'
    return response

@login_required
def eliminar_reporte(request, pk):
    if request.method == 'POST':
        reporte = get_object_or_404(Reporte, pk=pk, usuario_solicita=request.user)
        reporte.delete()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@login_required
def estadisticas_ventas(request):
    """API endpoint para estadísticas de ventas"""
    dias = int(request.GET.get('dias', 30))
    fecha_inicio = date.today() - timedelta(days=dias)
    
    estadisticas = EstadisticaVenta.objects.filter(
        fecha__gte=fecha_inicio
    ).order_by('fecha')
    
    data = {
        'fechas': [est.fecha.strftime('%Y-%m-%d') for est in estadisticas],
        'ventas': [float(est.total_ventas) for est in estadisticas],
        'cantidad': [est.cantidad_ventas for est in estadisticas],
    }
    
    return JsonResponse(data)

@login_required
def estadisticas_inventario(request):
    """API endpoint para estadísticas de inventario"""
    # Productos con bajo stock
    from django.db import models
    monturas_bajo = Montura.objects.filter(
        stock_actual__lte=models.F('stock_minimo'),
        activo=True
    ).count()
    
    cristales_bajo = Cristal.objects.filter(
        stock_actual__lte=models.F('stock_minimo'),
        activo=True
    ).count()
    
    # Valor del inventario
    valor_monturas = Montura.objects.filter(activo=True).aggregate(
        total=Sum(models.F('stock_actual') * models.F('precio_compra'))
    )['total'] or 0
    
    valor_cristales = Cristal.objects.filter(activo=True).aggregate(
        total=Sum(models.F('stock_actual') * models.F('precio_compra'))
    )['total'] or 0
    
    data = {
        'productos_bajo_stock': monturas_bajo + cristales_bajo,
        'valor_inventario': float(valor_monturas + valor_cristales),
        'monturas_total': Montura.objects.filter(activo=True).count(),
        'cristales_total': Cristal.objects.filter(activo=True).count(),
    }
    
    return JsonResponse(data)

@login_required
def estadisticas_taller(request):
    """API endpoint para estadísticas del taller"""
    ordenes = OrdenTrabajo.objects.all()
    
    data = {
        'pendientes': ordenes.filter(estado='pendiente').count(),
        'en_proceso': ordenes.filter(estado='en_proceso').count(),
        'terminadas': ordenes.filter(estado='terminada').count(),
        'atrasadas': ordenes.filter(
            fecha_fin_estimada__lt=timezone.now(),
            estado__in=['pendiente', 'en_proceso']
        ).count(),
    }
    
    return JsonResponse(data)

@login_required
def ventas_por_mes(request):
    """API para gráfico de ventas por mes"""
    from django.db.models.functions import TruncMonth
    
    ventas = Venta.objects.filter(
        fecha__gte=date.today() - timedelta(days=365),
        estado__in=['confirmada', 'entregada']
    ).annotate(
        mes=TruncMonth('fecha')
    ).values('mes').annotate(
        total=Sum('total'),
        cantidad=Count('id')
    ).order_by('mes')
    
    data = {
        'meses': [v['mes'].strftime('%Y-%m') for v in ventas],
        'totales': [float(v['total']) for v in ventas],
        'cantidades': [v['cantidad'] for v in ventas],
    }
    
    return JsonResponse(data)

@login_required
def productos_populares(request):
    """API para productos más vendidos"""
    # Monturas más vendidas
    monturas = ItemVenta.objects.filter(
        montura__isnull=False,
        venta__estado__in=['confirmada', 'entregada']
    ).values(
        'montura__nombre', 'montura__codigo'
    ).annotate(
        cantidad=Sum('cantidad')
    ).order_by('-cantidad')[:10]
    
    # Cristales más vendidos
    cristales = ItemVenta.objects.filter(
        cristal__isnull=False,
        venta__estado__in=['confirmada', 'entregada']
    ).values(
        'cristal__tipo_cristal__nombre', 'cristal__codigo'
    ).annotate(
        cantidad=Sum('cantidad')
    ).order_by('-cantidad')[:10]
    
    data = {
        'monturas': [
            {
                'nombre': m['montura__nombre'],
                'codigo': m['montura__codigo'],
                'cantidad': m['cantidad']
            } for m in monturas
        ],
        'cristales': [
            {
                'nombre': c['cristal__tipo_cristal__nombre'],
                'codigo': c['cristal__codigo'],
                'cantidad': c['cantidad']
            } for c in cristales
        ]
    }
    
    return JsonResponse(data)

@login_required
def estado_ordenes_taller(request):
    """API para estado de órdenes del taller"""
    estados = OrdenTrabajo.objects.values('estado').annotate(
        cantidad=Count('id')
    ).order_by('estado')
    
    data = {
        'estados': [e['estado'] for e in estados],
        'cantidades': [e['cantidad'] for e in estados],
        'labels': [dict(OrdenTrabajo.ESTADOS).get(e['estado'], e['estado']) for e in estados]
    }
    
    return JsonResponse(data)
