# 🛒 Tienda Online Cuba

Una plataforma de comercio electrónico moderna desarrollada con Django, diseñada específicamente para el mercado cubano con soporte para múltiples monedas (CUP, USD, EUR).

## ✨ Características

### 🔐 Sistema de Autenticación
- Registro e inicio de sesión de usuarios
- Perfiles de usuario personalizados
- Diferentes tipos de usuario (Cliente, Empleado, Administrador)
- Sistema de permisos integrado

### 📦 Gestión de Productos
- Catálogo completo de productos
- Categorías organizadas
- Múltiples imágenes por producto
- Información detallada (nombre, descripción, código, precios, stock)
- Soporte para múltiples monedas (CUP, USD, EUR)
- Control de inventario

### 🛠️ Panel de Administración
- Interfaz de administración completa
- Gestión de productos e inventario
- Control de categorías
- Gestión de usuarios
- Reportes de stock y márgenes de ganancia
- Acciones masivas para productos

### 🎨 Interfaz de Usuario
- Diseño moderno inspirado en Amazon
- Responsive design para móviles y desktop
- Búsqueda avanzada con filtros
- Paginación de productos
- Animaciones y efectos visuales
- Experiencia de usuario optimizada

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd tienda_online
   ```

2. **Crear entorno virtual**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Linux/Mac
   # o
   venv\Scripts\activate     # En Windows
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar base de datos**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Crear superusuario**
   ```bash
   python manage.py createsuperuser
   ```

6. **Ejecutar servidor de desarrollo**
   ```bash
   python manage.py runserver
   ```

7. **Acceder a la aplicación**
   - Sitio web: http://127.0.0.1:8000/
   - Panel admin: http://127.0.0.1:8000/admin/

## 📁 Estructura del Proyecto

```
tienda_online/
├── tienda_online/          # Configuración principal del proyecto
│   ├── settings.py         # Configuraciones de Django
│   ├── urls.py            # URLs principales
│   └── wsgi.py            # Configuración WSGI
├── productos/             # Aplicación de productos
│   ├── models.py          # Modelos de productos y categorías
│   ├── views.py           # Vistas de productos
│   ├── admin.py           # Configuración del admin
│   └── urls.py            # URLs de productos
├── usuarios/              # Aplicación de usuarios
│   ├── models.py          # Modelo de perfil de usuario
│   ├── views.py           # Vistas de autenticación
│   ├── admin.py           # Configuración del admin
│   └── urls.py            # URLs de usuarios
├── templates/             # Plantillas HTML
│   ├── base.html          # Plantilla base
│   ├── productos/         # Plantillas de productos
│   └── usuarios/          # Plantillas de usuarios
├── static/                # Archivos estáticos
│   ├── css/               # Estilos CSS
│   ├── js/                # JavaScript
│   └── img/               # Imágenes
├── media/                 # Archivos subidos por usuarios
├── requirements.txt       # Dependencias del proyecto
└── README.md             # Este archivo
```

## 🔧 Configuración

### Variables de Entorno
Para producción, es recomendable usar variables de entorno:

```bash
# Crear archivo .env
SECRET_KEY=tu-clave-secreta-aqui
DEBUG=False
ALLOWED_HOSTS=tu-dominio.com
```

### Base de Datos
Por defecto usa SQLite para desarrollo. Para producción, configura PostgreSQL:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'tienda_online',
        'USER': 'tu_usuario',
        'PASSWORD': 'tu_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

## 📊 Modelos de Datos

### Producto
- Nombre, descripción, código único
- Precios de compra y venta
- Moneda (CUP, USD, EUR)
- Stock y stock mínimo
- Imágenes múltiples
- Categoría, marca, dimensiones
- Estado (disponible, agotado, descontinuado)

### Categoría
- Nombre único
- Descripción e imagen
- Estado activo/inactivo

### Perfil de Usuario
- Extensión del modelo User de Django
- Tipo de usuario (Cliente, Empleado, Admin)
- Información de contacto y dirección
- Avatar personalizado

## 🛡️ Seguridad

- Validación CSRF en todos los formularios
- Autenticación requerida para acciones sensibles
- Validación de datos en el servidor
- Protección contra inyección SQL
- Configuración segura por defecto

## 🌐 Características Específicas para Cuba

- **Monedas**: Soporte nativo para CUP, USD y EUR
- **Zona horaria**: Configurada para America/Havana
- **Idioma**: Interfaz en español
- **Direcciones**: Campos adaptados para direcciones cubanas
- **Contacto**: Formatos de teléfono cubanos

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Si tienes problemas o preguntas:

1. Revisa la documentación
2. Busca en los issues existentes
3. Crea un nuevo issue con detalles del problema

## 🚀 Roadmap

### Próximas características:
- [ ] Sistema de carrito de compras
- [ ] Proceso de checkout
- [ ] Integración con pasarelas de pago
- [ ] Sistema de reviews y calificaciones
- [ ] Notificaciones por email
- [ ] API REST
- [ ] App móvil
- [ ] Sistema de cupones y descuentos
- [ ] Reportes avanzados
- [ ] Integración con redes sociales

## 📞 Contacto

- **Email**: info@tiendacuba.com
- **Teléfono**: +53 5555-5555
- **Dirección**: La Habana, Cuba

---

Desarrollado con ❤️ para el mercado cubano 