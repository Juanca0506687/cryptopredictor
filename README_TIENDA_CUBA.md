# 🛍️ Tienda Cuba - E-commerce con Django

Una aplicación web completa de comercio electrónico desarrollada con Django, diseñada específicamente para el mercado cubano con soporte para múltiples monedas y un panel de administración completo.

## 🌟 Características Principales

### 🔐 Sistema de Autenticación
- Registro e inicio de sesión de usuarios
- Perfiles de usuario extendidos
- Diferentes tipos de usuario (Cliente, Vendedor, Administrador)
- Sistema de permisos integrado

### 📦 Gestión de Productos
- Catálogo completo de productos
- Categorización avanzada
- Múltiples imágenes por producto
- Control de inventario en tiempo real
- Sistema de productos destacados
- Búsqueda y filtrado avanzado

### 💰 Soporte Multi-Moneda
- **CUP** (Peso Cubano)
- **USD** (Dólar Americano)
- **EUR** (Euro)
- **MLC** (Moneda Libremente Convertible)

### 🎨 Diseño Moderno
- Interfaz estilo Amazon
- Diseño responsive (móvil y desktop)
- Bootstrap 5 integrado
- Iconos Font Awesome
- Experiencia de usuario optimizada

### 🔧 Panel de Administración
- Gestión completa de productos
- Control de inventario
- Estadísticas y métricas
- Gestión de categorías
- Administración de usuarios

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- pip (gestor de paquetes de Python)

### Instalación

1. **Clonar el repositorio**
```bash
cd /workspace
```

2. **Instalar dependencias**
```bash
pip install --break-system-packages django pillow
```

3. **Configurar la base de datos**
```bash
export PATH=$PATH:/home/ubuntu/.local/bin
python3 manage.py makemigrations
python3 manage.py migrate
```

4. **Crear superusuario**
```bash
python3 manage.py createsuperuser
```
O usar el usuario predeterminado:
- **Usuario:** admin
- **Contraseña:** admin123
- **Email:** admin@tiendacuba.cu

5. **Cargar datos de ejemplo**
```bash
python3 crear_datos_ejemplo.py
```

6. **Ejecutar el servidor**
```bash
python3 manage.py runserver 0.0.0.0:8000
```

## 📁 Estructura del Proyecto

```
tienda_cuba/
├── tienda_cuba/          # Configuración principal
│   ├── settings.py       # Configuraciones
│   ├── urls.py          # URLs principales
│   └── wsgi.py          # WSGI config
├── productos/           # App de productos
│   ├── models.py        # Modelos de productos y categorías
│   ├── views.py         # Vistas del catálogo
│   ├── admin.py         # Configuración del admin
│   └── urls.py          # URLs de productos
├── usuarios/            # App de usuarios
│   ├── models.py        # Perfil de usuario extendido
│   ├── views.py         # Autenticación y perfil
│   ├── admin.py         # Admin de usuarios
│   └── urls.py          # URLs de usuarios
├── templates/           # Templates HTML
│   ├── base.html        # Template base
│   ├── productos/       # Templates de productos
│   └── usuarios/        # Templates de usuarios
├── static/              # Archivos estáticos
├── media/               # Archivos multimedia
└── manage.py           # Script de gestión Django
```

## 🎯 Funcionalidades Implementadas

### Para Usuarios
- ✅ Registro e inicio de sesión
- ✅ Navegación del catálogo
- ✅ Búsqueda y filtrado de productos
- ✅ Vista detallada de productos
- ✅ Perfil de usuario
- ✅ Navegación por categorías

### Para Administradores
- ✅ Panel de administración completo
- ✅ Gestión de productos (CRUD completo)
- ✅ Control de inventario
- ✅ Gestión de categorías
- ✅ Administración de usuarios
- ✅ Estadísticas y métricas
- ✅ Carga masiva de imágenes

## 🛠️ Modelos de Datos

### Producto
- Información básica (nombre, descripción, código)
- Precios (compra, venta) con moneda
- Control de inventario (stock, stock mínimo)
- Imágenes múltiples
- Categorización
- Estado y fechas
- Métricas (vistas, ventas)

### Categoría
- Nombre y descripción
- Estado activo/inactivo
- Fecha de creación

### Perfil de Usuario
- Información personal extendida
- Tipo de usuario
- Fecha de registro
- Estado activo

## 🎨 Características de Diseño

### Colores Principales
- **Primario:** #232f3e (Azul oscuro estilo Amazon)
- **Secundario:** #ff9900 (Naranja Amazon)
- **Acento:** #146eb4 (Azul)

### Componentes UI
- Cards responsivas para productos
- Navegación sticky
- Breadcrumbs
- Paginación
- Filtros dinámicos
- Badges de estado
- Modales y alertas

## 📊 Panel de Administración

Accede al panel en: `http://localhost:8000/admin/`

### Características del Admin
- **Productos:** Gestión completa con vista previa de imágenes
- **Categorías:** Control de categorías activas
- **Usuarios:** Administración de perfiles
- **Estadísticas:** Métricas en tiempo real
- **Acciones masivas:** Operaciones en lote

## 🔧 Configuración Avanzada

### Configuración de Media Files
```python
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
```

### Configuración Regional
```python
LANGUAGE_CODE = 'es-es'
TIME_ZONE = 'America/Havana'
```

### Configuración de Autenticación
```python
LOGIN_URL = '/usuarios/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'
```

## 🚀 Despliegue

### Variables de Entorno Recomendadas
```bash
DEBUG=False
SECRET_KEY=tu-clave-secreta-super-segura
ALLOWED_HOSTS=tu-dominio.com
```

### Para Producción
1. Configurar base de datos PostgreSQL
2. Configurar servidor web (Nginx/Apache)
3. Configurar WSGI (Gunicorn)
4. Configurar archivos estáticos
5. Configurar SSL/HTTPS

## 📈 Datos de Ejemplo

El proyecto incluye un script para cargar datos de ejemplo:

### Categorías Incluidas
- Electrónicos
- Ropa y Accesorios  
- Hogar y Jardín
- Deportes
- Libros
- Alimentación

### Productos de Ejemplo
- iPhone 13, Samsung Galaxy S21
- Laptop HP Pavilion
- Ropa (Polos, Jeans)
- Electrodomésticos (Cafetera)
- Artículos deportivos
- Libros y alimentos

## 🤝 Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Realiza tus cambios
4. Envía un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Para soporte técnico o preguntas:
- Email: info@tiendacuba.cu
- Teléfono: +53 7 123-4567

## 🎉 Características Futuras

### Próximas Implementaciones
- [ ] Carrito de compras
- [ ] Sistema de pagos
- [ ] Wishlist/Lista de deseos
- [ ] Reviews y calificaciones
- [ ] Sistema de cupones
- [ ] Notificaciones por email
- [ ] API REST
- [ ] App móvil

---

**Desarrollado con ❤️ para Cuba** 🇨🇺