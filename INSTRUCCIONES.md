# 🎉 ¡Tu Tienda Online Cuba está Lista!

## 🚀 Cómo Ejecutar la Aplicación

### 1. Activar el Entorno Virtual
```bash
cd tienda_online
source ../venv/bin/activate
```

### 2. Ejecutar el Servidor
```bash
python manage.py runserver
```

### 3. Acceder a la Aplicación
- **Sitio Web**: http://127.0.0.1:8000/
- **Panel Admin**: http://127.0.0.1:8000/admin/

## 👤 Usuarios Creados

### Administradores
- **Usuario**: `admin2`
- **Contraseña**: `admin123`
- **Email**: admin2@tiendacuba.com

### Primer Administrador (sin contraseña asignada)
- **Usuario**: `admin`
- **Email**: admin@tiendacuba.com
- **Nota**: Necesitarás cambiar la contraseña desde el panel admin

## 📋 Primeros Pasos

### 1. Acceder al Panel de Administración
1. Ve a http://127.0.0.1:8000/admin/
2. Inicia sesión con `admin2` / `admin123`
3. Cambia la contraseña por una más segura

### 2. Crear Categorías
1. En el admin, ve a "Categorías"
2. Crea categorías como:
   - Electrónicos
   - Ropa y Accesorios
   - Hogar y Jardín
   - Deportes
   - Libros y Educación

### 3. Agregar Productos
1. Ve a "Productos" en el admin
2. Haz clic en "Agregar producto"
3. Completa todos los campos:
   - Nombre del producto
   - Descripción detallada
   - Código único
   - Categoría
   - Precios (compra y venta)
   - Moneda (CUP, USD, EUR)
   - Stock
   - Imágenes

### 4. Configurar Productos Destacados
1. En la lista de productos
2. Marca algunos como "destacado"
3. Estos aparecerán en la página principal

## 🌟 Características Principales

### ✅ Sistema de Autenticación
- Registro de usuarios
- Inicio de sesión
- Perfiles de usuario
- Diferentes tipos de usuario

### ✅ Gestión de Productos
- Catálogo completo
- Búsqueda y filtros
- Múltiples imágenes
- Control de inventario
- Soporte para CUP, USD, EUR

### ✅ Panel de Administración
- Gestión completa de productos
- Control de categorías
- Gestión de usuarios
- Reportes de stock
- Acciones masivas

### ✅ Interfaz Moderna
- Diseño responsive
- Estilo Amazon
- Animaciones suaves
- Experiencia optimizada

## 🛠️ Personalización

### Cambiar el Logo
1. Reemplaza el texto "TiendaCuba" en `templates/base.html`
2. Agrega tu logo en `static/img/`

### Modificar Colores
1. Edita `static/css/style.css`
2. Cambia las variables de color

### Agregar Funcionalidades
1. Carrito de compras
2. Sistema de pagos
3. Envío de emails
4. Reportes avanzados

## 📞 Configuración de Contacto

Edita en `templates/base.html`:
- Teléfono: Línea 152
- Email: Línea 153
- Redes sociales: Líneas 158-160

## 🔧 Configuración Avanzada

### Base de Datos en Producción
Para usar PostgreSQL en lugar de SQLite:

```python
# En settings.py
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

### Variables de Entorno
Crea un archivo `.env`:

```
SECRET_KEY=tu-clave-secreta-aqui
DEBUG=False
ALLOWED_HOSTS=tu-dominio.com
```

### Archivos Estáticos en Producción
```bash
python manage.py collectstatic
```

## 🚀 Despliegue

### Heroku
1. Instala Heroku CLI
2. `heroku create tu-app`
3. `git push heroku main`

### Railway
1. Conecta tu repositorio GitHub
2. Railway detectará Django automáticamente

### DigitalOcean/AWS
1. Configura un servidor
2. Instala nginx + gunicorn
3. Configura SSL

## 📚 Próximos Pasos

1. **Carrito de Compras**: Implementar funcionalidad de carrito
2. **Pagos**: Integrar pasarelas de pago
3. **Emails**: Configurar notificaciones por email
4. **Reviews**: Sistema de reseñas de productos
5. **API**: Crear API REST para app móvil
6. **Reportes**: Dashboard con estadísticas

## 🆘 Soporte

Si tienes problemas:

1. Verifica que el entorno virtual esté activado
2. Asegúrate de que todas las dependencias estén instaladas
3. Revisa los logs del servidor para errores
4. Consulta la documentación de Django

## 🎊 ¡Felicidades!

Has creado una tienda online completa y moderna. Tu plataforma incluye:

- ✅ Sistema de usuarios completo
- ✅ Gestión de inventario profesional
- ✅ Interfaz moderna y responsive
- ✅ Panel de administración potente
- ✅ Soporte para múltiples monedas
- ✅ Optimizada para el mercado cubano

¡Ahora puedes empezar a agregar productos y hacer crecer tu negocio online! 🚀