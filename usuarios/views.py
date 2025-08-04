from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, TemplateView
from django.contrib import messages
from django.urls import reverse_lazy
from django.http import JsonResponse
from .models import Usuario
from .forms import UsuarioCreationForm, UsuarioUpdateForm, PerfilForm

class CustomLoginView(LoginView):
    template_name = 'usuarios/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('reportes:dashboard')

class AdminRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_authenticated and self.request.user.puede_administrar()

class UsuarioListView(AdminRequiredMixin, ListView):
    model = Usuario
    template_name = 'usuarios/lista.html'
    context_object_name = 'usuarios'
    paginate_by = 20
    
    def get_queryset(self):
        return Usuario.objects.all().order_by('username')

class UsuarioDetailView(AdminRequiredMixin, DetailView):
    model = Usuario
    template_name = 'usuarios/detalle.html'
    context_object_name = 'usuario'

class UsuarioCreateView(AdminRequiredMixin, CreateView):
    model = Usuario
    form_class = UsuarioCreationForm
    template_name = 'usuarios/crear.html'
    success_url = reverse_lazy('usuarios:lista')
    
    def form_valid(self, form):
        messages.success(self.request, 'Usuario creado exitosamente.')
        return super().form_valid(form)

class UsuarioUpdateView(AdminRequiredMixin, UpdateView):
    model = Usuario
    form_class = UsuarioUpdateForm
    template_name = 'usuarios/editar.html'
    success_url = reverse_lazy('usuarios:lista')
    
    def form_valid(self, form):
        messages.success(self.request, 'Usuario actualizado exitosamente.')
        return super().form_valid(form)

class PerfilView(LoginRequiredMixin, TemplateView):
    template_name = 'usuarios/perfil.html'

class EditarPerfilView(LoginRequiredMixin, UpdateView):
    model = Usuario
    form_class = PerfilForm
    template_name = 'usuarios/editar_perfil.html'
    success_url = reverse_lazy('usuarios:perfil')
    
    def get_object(self):
        return self.request.user
    
    def form_valid(self, form):
        messages.success(self.request, 'Perfil actualizado exitosamente.')
        return super().form_valid(form)

@login_required
@user_passes_test(lambda u: u.puede_administrar())
def toggle_usuario_activo(request, pk):
    if request.method == 'POST':
        usuario = get_object_or_404(Usuario, pk=pk)
        usuario.activo = not usuario.activo
        usuario.save()
        
        estado = 'activado' if usuario.activo else 'desactivado'
        messages.success(request, f'Usuario {estado} exitosamente.')
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'activo': usuario.activo,
                'mensaje': f'Usuario {estado} exitosamente.'
            })
    
    return redirect('usuarios:lista')
