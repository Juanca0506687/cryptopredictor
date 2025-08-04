#!/bin/bash

# Script de instalación para Sistema de Gestión Óptica MIPYME
# Desarrollado para MIPYMES cubanas del sector óptico

echo "🔍 ====================================="
echo "   Sistema de Gestión Óptica MIPYME   "
echo "===================================== 🔍"
echo ""
echo "Iniciando instalación..."
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar mensajes
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[ÉXITO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[ADVERTENCIA]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar si Python está instalado
print_status "Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION encontrado"
else
    print_error "Python 3 no está instalado. Por favor instálelo primero."
    exit 1
fi

# Verificar si pip está instalado
print_status "Verificando pip..."
if command -v pip3 &> /dev/null; then
    print_success "pip encontrado"
else
    print_error "pip no está instalado. Instalando..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
fi

# Crear entorno virtual (opcional)
read -p "¿Desea crear un entorno virtual? (recomendado) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    print_status "Creando entorno virtual..."
    python3 -m venv venv_optica
    source venv_optica/bin/activate
    print_success "Entorno virtual creado y activado"
else
    print_warning "Continuando sin entorno virtual"
fi

# Instalar dependencias
print_status "Instalando dependencias de Django..."
if [ -f "requirements_django.txt" ]; then
    pip3 install -r requirements_django.txt
    if [ $? -eq 0 ]; then
        print_success "Dependencias instaladas correctamente"
    else
        print_error "Error al instalar dependencias"
        exit 1
    fi
else
    print_warning "Archivo requirements_django.txt no encontrado. Instalando Django manualmente..."
    pip3 install Django==4.2.7
fi

# Verificar instalación de Django
print_status "Verificando instalación de Django..."
python3 -c "import django; print('Django', django.get_version())" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Django instalado correctamente"
else
    print_error "Error en la instalación de Django"
    exit 1
fi

# Crear directorios necesarios
print_status "Creando directorios necesarios..."
mkdir -p media
mkdir -p static
mkdir -p staticfiles
print_success "Directorios creados"

# Realizar migraciones
print_status "Configurando base de datos..."
python3 manage.py makemigrations
if [ $? -eq 0 ]; then
    print_success "Migraciones creadas"
else
    print_error "Error al crear migraciones"
    exit 1
fi

python3 manage.py migrate
if [ $? -eq 0 ]; then
    print_success "Base de datos configurada"
else
    print_error "Error al aplicar migraciones"
    exit 1
fi

# Crear superusuario automáticamente
print_status "Creando usuario administrador..."
python3 manage.py shell << EOF
from usuarios.models import Usuario
try:
    if not Usuario.objects.filter(username='admin').exists():
        Usuario.objects.create_superuser('admin', 'admin@optica.cu', 'admin123', rol='administrador')
        print("Usuario administrador creado: admin/admin123")
    else:
        print("Usuario administrador ya existe")
except Exception as e:
    print(f"Error: {e}")
EOF

print_success "Usuario administrador configurado"

# Recopilar archivos estáticos
print_status "Recopilando archivos estáticos..."
python3 manage.py collectstatic --noinput
print_success "Archivos estáticos configurados"

# Información final
echo ""
echo "🎉 ¡INSTALACIÓN COMPLETADA! 🎉"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  INFORMACIÓN DE ACCESO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 URL del Sistema: http://localhost:8000"
echo "👤 Usuario: admin"
echo "🔑 Contraseña: admin123"
echo "⚙️  Panel Admin: http://localhost:8000/admin/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  COMANDOS ÚTILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Iniciar servidor:"
echo "   python3 manage.py runserver"
echo ""
echo "🚀 Iniciar en red local:"
echo "   python3 manage.py runserver 0.0.0.0:8000"
echo ""
echo "💾 Crear backup:"
echo "   python3 manage.py dumpdata > backup_\$(date +%Y%m%d).json"
echo ""
echo "📦 Crear nuevo usuario:"
echo "   python3 manage.py createsuperuser"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PRÓXIMOS PASOS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Ejecutar: python3 manage.py runserver"
echo "2. Abrir navegador en: http://localhost:8000"
echo "3. Iniciar sesión con admin/admin123"
echo "4. Crear usuarios para vendedores y técnicos"
echo "5. Configurar productos y categorías"
echo "6. ¡Comenzar a usar el sistema!"
echo ""
echo "📖 Para más información, consulte README_OPTICA.md"
echo ""

# Preguntar si desea iniciar el servidor
read -p "¿Desea iniciar el servidor ahora? [y/N]: " start_server
if [[ $start_server =~ ^[Yy]$ ]]; then
    print_status "Iniciando servidor..."
    echo ""
    echo "🚀 Servidor iniciado en http://localhost:8000"
    echo "   Presione Ctrl+C para detener el servidor"
    echo ""
    python3 manage.py runserver
fi

echo ""
echo "¡Gracias por usar el Sistema de Gestión Óptica MIPYME! 🔍👓"