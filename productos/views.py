from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from django.db.models import Q
from .models import Producto, Categoria

# Create your views here.

def home(request):
    """Página principal con productos destacados"""
    productos_destacados = Producto.objects.filter(destacado=True, estado='activo')[:8]
    categorias = Categoria.objects.filter(activa=True)[:6]
    productos_recientes = Producto.objects.filter(estado='activo').order_by('-fecha_creacion')[:6]
    
    context = {
        'productos_destacados': productos_destacados,
        'categorias': categorias,
        'productos_recientes': productos_recientes,
    }
    return render(request, 'productos/home.html', context)

def catalogo(request):
    """Catálogo de productos con filtros y búsqueda"""
    productos = Producto.objects.filter(estado='activo')
    categorias = Categoria.objects.filter(activa=True)
    
    # Filtros
    categoria_id = request.GET.get('categoria')
    busqueda = request.GET.get('q')
    moneda = request.GET.get('moneda')
    orden = request.GET.get('orden', 'nombre')
    
    if categoria_id:
        productos = productos.filter(categoria_id=categoria_id)
    
    if busqueda:
        productos = productos.filter(
            Q(nombre__icontains=busqueda) |
            Q(descripcion__icontains=busqueda) |
            Q(codigo__icontains=busqueda)
        )
    
    if moneda:
        productos = productos.filter(moneda=moneda)
    
    # Ordenamiento
    if orden == 'precio_asc':
        productos = productos.order_by('precio_venta')
    elif orden == 'precio_desc':
        productos = productos.order_by('-precio_venta')
    elif orden == 'fecha':
        productos = productos.order_by('-fecha_creacion')
    else:
        productos = productos.order_by('nombre')
    
    # Paginación
    paginator = Paginator(productos, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'categorias': categorias,
        'categoria_actual': categoria_id,
        'busqueda': busqueda,
        'moneda_actual': moneda,
        'orden_actual': orden,
    }
    return render(request, 'productos/catalogo.html', context)

def producto_detalle(request, pk):
    """Detalle de un producto específico"""
    producto = get_object_or_404(Producto, pk=pk, estado='activo')
    producto.incrementar_vistas()
    
    # Productos relacionados de la misma categoría
    productos_relacionados = Producto.objects.filter(
        categoria=producto.categoria,
        estado='activo'
    ).exclude(pk=producto.pk)[:4]
    
    context = {
        'producto': producto,
        'productos_relacionados': productos_relacionados,
    }
    return render(request, 'productos/detalle.html', context)

def categoria_detalle(request, pk):
    """Productos de una categoría específica"""
    categoria = get_object_or_404(Categoria, pk=pk, activa=True)
    productos = Producto.objects.filter(categoria=categoria, estado='activo')
    
    # Paginación
    paginator = Paginator(productos, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'categoria': categoria,
        'page_obj': page_obj,
    }
    return render(request, 'productos/categoria.html', context)
