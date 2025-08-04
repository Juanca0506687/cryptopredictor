from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q
from .models import Usuario, PerfilUsuario, SesionUsuario
from .forms import LoginForm, UsuarioForm, PerfilUsuarioForm
import json


def es_administrador(user):
    """Verificar si el usuario es administrador"""
    return user.is_authenticated and user.es_administrador()


def login_view(request):
    """Vista de login"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            user = authenticate(request, username=username, password=password)
            if user is not None and user.activo:
                login(request, user)
                
                # Registrar sesión
                SesionUsuario.objects.create(
                    usuario=user,
                    ip_address=get_client_ip(request),
                    user_agent=request.META.get('HTTP_USER_AGENT', '')
                )
                
                messages.success(request, f'¡Bienvenido {user.first_name or user.username}!')
                return redirect('dashboard')
            else:
                messages.error(request, 'Usuario o contraseña incorrectos, o usuario inactivo.')
    else:
        form = LoginForm()
    
    return render(request, 'usuarios/login.html', {'form': form})


@login_required
def logout_view(request):
    """Vista de logout"""
    # Marcar sesión como terminada
    try:
        sesion = SesionUsuario.objects.filter(
            usuario=request.user,
            activa=True
        ).latest('fecha_inicio')
        sesion.activa = False
        sesion.save()
    except SesionUsuario.DoesNotExist:
        pass
    
    logout(request)
    messages.info(request, 'Has cerrado sesión correctamente.')
    return redirect('login')


@login_required
def dashboard(request):
    """Dashboard principal según el rol del usuario"""
    context = {
        'usuario': request.user,
    }
    
    if request.user.es_administrador():
        # Estadísticas para administrador
        context.update({
            'total_usuarios': Usuario.objects.filter(activo=True).count(),
            'usuarios_conectados': SesionUsuario.objects.filter(activa=True).count(),
        })
        template = 'usuarios/dashboard_admin.html'
    elif request.user.es_vendedor():
        template = 'usuarios/dashboard_vendedor.html'
    elif request.user.es_tecnico():
        template = 'usuarios/dashboard_tecnico.html'
    else:
        template = 'usuarios/dashboard.html'
    
    return render(request, template, context)


@login_required
@user_passes_test(es_administrador)
def lista_usuarios(request):
    """Lista de usuarios - solo para administradores"""
    usuarios = Usuario.objects.all()
    
    # Filtros
    rol_filter = request.GET.get('rol')
    activo_filter = request.GET.get('activo')
    buscar = request.GET.get('buscar')
    
    if rol_filter:
        usuarios = usuarios.filter(rol=rol_filter)
    
    if activo_filter:
        usuarios = usuarios.filter(activo=activo_filter == 'true')
    
    if buscar:
        usuarios = usuarios.filter(
            Q(username__icontains=buscar) |
            Q(first_name__icontains=buscar) |
            Q(last_name__icontains=buscar) |
            Q(email__icontains=buscar) |
            Q(cedula__icontains=buscar)
        )
    
    paginator = Paginator(usuarios, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'roles': Usuario.ROLES_CHOICES,
        'filtros': {
            'rol': rol_filter,
            'activo': activo_filter,
            'buscar': buscar,
        }
    }
    
    return render(request, 'usuarios/lista_usuarios.html', context)


@login_required
@user_passes_test(es_administrador)
def crear_usuario(request):
    """Crear nuevo usuario - solo para administradores"""
    if request.method == 'POST':
        form = UsuarioForm(request.POST)
        if form.is_valid():
            usuario = form.save()
            messages.success(request, f'Usuario {usuario.username} creado exitosamente.')
            return redirect('lista_usuarios')
    else:
        form = UsuarioForm()
    
    return render(request, 'usuarios/crear_usuario.html', {'form': form})


@login_required
def perfil_usuario(request):
    """Ver y editar perfil del usuario actual"""
    try:
        perfil = request.user.perfil
    except PerfilUsuario.DoesNotExist:
        perfil = PerfilUsuario.objects.create(usuario=request.user)
    
    if request.method == 'POST':
        form = PerfilUsuarioForm(request.POST, request.FILES, instance=perfil)
        if form.is_valid():
            form.save()
            messages.success(request, 'Perfil actualizado exitosamente.')
            return redirect('perfil_usuario')
    else:
        form = PerfilUsuarioForm(instance=perfil)
    
    return render(request, 'usuarios/perfil.html', {'form': form, 'perfil': perfil})


@login_required
@user_passes_test(es_administrador)
def toggle_usuario_activo(request, usuario_id):
    """Activar/desactivar usuario - solo para administradores"""
    if request.method == 'POST':
        usuario = get_object_or_404(Usuario, id=usuario_id)
        usuario.activo = not usuario.activo
        usuario.save()
        
        estado = 'activado' if usuario.activo else 'desactivado'
        messages.success(request, f'Usuario {usuario.username} {estado} exitosamente.')
        
        return JsonResponse({'success': True, 'activo': usuario.activo})
    
    return JsonResponse({'success': False})


@login_required
@user_passes_test(es_administrador)
def sesiones_activas(request):
    """Ver sesiones activas - solo para administradores"""
    sesiones = SesionUsuario.objects.filter(activa=True).select_related('usuario')
    
    return render(request, 'usuarios/sesiones_activas.html', {'sesiones': sesiones})


def get_client_ip(request):
    """Obtener IP del cliente"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
