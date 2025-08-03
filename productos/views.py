from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from django.db.models import Q
from .models import Producto, Categoria

def home_view(request):
    """Vista principal con productos destacados"""
    productos_destacados = Producto.objects.filter(
        activo=True, destacado=True, estado='disponible'
    )[:8]
    categorias = Categoria.objects.filter(activa=True)[:6]
    
    context = {
        'productos_destacados': productos_destacados,
        'categorias': categorias,
    }
    return render(request, 'productos/home.html', context)

def catalogo_view(request):
    """Vista del catálogo de productos con filtros"""
    productos = Producto.objects.filter(activo=True, estado='disponible')
    categorias = Categoria.objects.filter(activa=True)
    
    # Filtros
    categoria_id = request.GET.get('categoria')
    busqueda = request.GET.get('q')
    ordenar = request.GET.get('ordenar', 'nombre')
    
    if categoria_id:
        productos = productos.filter(categoria_id=categoria_id)
    
    if busqueda:
        productos = productos.filter(
            Q(nombre__icontains=busqueda) |
            Q(descripcion__icontains=busqueda) |
            Q(codigo__icontains=busqueda)
        )
    
    # Ordenamiento
    if ordenar == 'precio_asc':
        productos = productos.order_by('precio_venta')
    elif ordenar == 'precio_desc':
        productos = productos.order_by('-precio_venta')
    elif ordenar == 'nombre':
        productos = productos.order_by('nombre')
    elif ordenar == 'fecha':
        productos = productos.order_by('-fecha_creacion')
    
    # Paginación
    paginator = Paginator(productos, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'categorias': categorias,
        'categoria_seleccionada': categoria_id,
        'busqueda': busqueda,
        'ordenar': ordenar,
    }
    return render(request, 'productos/catalogo.html', context)

def producto_detalle_view(request, pk):
    """Vista de detalle de un producto"""
    producto = get_object_or_404(Producto, pk=pk, activo=True)
    productos_relacionados = Producto.objects.filter(
        categoria=producto.categoria,
        activo=True,
        estado='disponible'
    ).exclude(pk=producto.pk)[:4]
    
    context = {
        'producto': producto,
        'productos_relacionados': productos_relacionados,
    }
    return render(request, 'productos/detalle.html', context)

def categoria_view(request, pk):
    """Vista de productos por categoría"""
    categoria = get_object_or_404(Categoria, pk=pk, activa=True)
    productos = Producto.objects.filter(
        categoria=categoria,
        activo=True,
        estado='disponible'
    )
    
    # Paginación
    paginator = Paginator(productos, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'categoria': categoria,
        'page_obj': page_obj,
    }
    return render(request, 'productos/categoria.html', context)