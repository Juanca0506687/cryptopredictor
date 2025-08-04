from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError
from .models import Usuario, PerfilUsuario
import re


class LoginForm(forms.Form):
    """Formulario de login"""
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Usuario',
            'autocomplete': 'username'
        }),
        label='Usuario'
    )
    
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contraseña',
            'autocomplete': 'current-password'
        }),
        label='Contraseña'
    )


class UsuarioForm(UserCreationForm):
    """Formulario para crear y editar usuarios"""
    
    class Meta:
        model = Usuario
        fields = (
            'username', 'email', 'first_name', 'last_name',
            'rol', 'telefono', 'cedula', 'direccion', 'activo'
        )
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'rol': forms.Select(attrs={'class': 'form-select'}),
            'telefono': forms.TextInput(attrs={'class': 'form-control'}),
            'cedula': forms.TextInput(attrs={'class': 'form-control'}),
            'direccion': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'activo': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hacer campos obligatorios
        self.fields['email'].required = True
        self.fields['first_name'].required = True
        self.fields['last_name'].required = True
        
        # Añadir clases CSS a los campos de contraseña
        self.fields['password1'].widget.attrs.update({'class': 'form-control'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control'})
        
        # Personalizar labels
        self.fields['password1'].label = 'Contraseña'
        self.fields['password2'].label = 'Confirmar Contraseña'
    
    def clean_cedula(self):
        """Validar formato de cédula cubana"""
        cedula = self.cleaned_data.get('cedula')
        if cedula:
            # Formato básico de cédula cubana: 11 dígitos
            if not re.match(r'^\d{11}$', cedula):
                raise ValidationError('La cédula debe tener 11 dígitos.')
        return cedula
    
    def clean_telefono(self):
        """Validar formato de teléfono"""
        telefono = self.cleaned_data.get('telefono')
        if telefono:
            # Permitir diferentes formatos de teléfono
            if not re.match(r'^[\d\-\+\(\)\s]+$', telefono):
                raise ValidationError('Formato de teléfono inválido.')
        return telefono


class PerfilUsuarioForm(forms.ModelForm):
    """Formulario para editar perfil de usuario"""
    
    class Meta:
        model = PerfilUsuario
        fields = (
            'foto', 'biografia', 'fecha_nacimiento',
            'salario', 'comision_ventas'
        )
        widgets = {
            'biografia': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Cuéntanos un poco sobre ti...'
            }),
            'fecha_nacimiento': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date'
            }),
            'foto': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'salario': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.01',
                'min': '0'
            }),
            'comision_ventas': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.01',
                'min': '0',
                'max': '100'
            }),
        }
    
    def clean_foto(self):
        """Validar tamaño y tipo de imagen"""
        foto = self.cleaned_data.get('foto')
        if foto:
            # Verificar tamaño (máximo 5MB)
            if foto.size > 5 * 1024 * 1024:
                raise ValidationError('La imagen no puede ser mayor a 5MB.')
            
            # Verificar tipo de archivo
            if not foto.content_type.startswith('image/'):
                raise ValidationError('El archivo debe ser una imagen.')
        
        return foto
    
    def clean_comision_ventas(self):
        """Validar que la comisión esté entre 0 y 100"""
        comision = self.cleaned_data.get('comision_ventas')
        if comision is not None:
            if comision < 0 or comision > 100:
                raise ValidationError('La comisión debe estar entre 0% y 100%.')
        return comision


class CambiarContrasenaForm(forms.Form):
    """Formulario para cambiar contraseña"""
    contrasena_actual = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contraseña actual'
        }),
        label='Contraseña Actual'
    )
    
    nueva_contrasena = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nueva contraseña'
        }),
        label='Nueva Contraseña',
        min_length=8
    )
    
    confirmar_contrasena = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirmar nueva contraseña'
        }),
        label='Confirmar Nueva Contraseña'
    )
    
    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)
    
    def clean_contrasena_actual(self):
        """Verificar que la contraseña actual sea correcta"""
        contrasena_actual = self.cleaned_data.get('contrasena_actual')
        if not self.user.check_password(contrasena_actual):
            raise ValidationError('La contraseña actual es incorrecta.')
        return contrasena_actual
    
    def clean(self):
        """Verificar que las contraseñas nuevas coincidan"""
        cleaned_data = super().clean()
        nueva_contrasena = cleaned_data.get('nueva_contrasena')
        confirmar_contrasena = cleaned_data.get('confirmar_contrasena')
        
        if nueva_contrasena and confirmar_contrasena:
            if nueva_contrasena != confirmar_contrasena:
                raise ValidationError('Las contraseñas nuevas no coinciden.')
        
        return cleaned_data