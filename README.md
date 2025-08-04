# 🔧 Óptica MIPYME - Sistema de Gestión Integral

Sistema completo de gestión para una MIPYME de venta de espejuelos graduados en Cuba, desarrollado con Django. Incluye gestión de productos, ventas, clientes, taller y reportes con funcionalidad offline.

## ✨ Características Principales

### 🔐 Sistema Multiusuario
- **Administrador**: Acceso completo al sistema
- **Vendedor**: Gestión de ventas y clientes
- **Técnico**: Gestión del taller y órdenes de trabajo

### 📦 Gestión de Productos
- **Monturas**: Registro con foto, precio, stock y características
- **Cristales**: Diferentes graduaciones y tratamientos
- **Control de inventario** con alertas de bajo stock
- **Movimientos de inventario** automáticos

### 💰 Sistema de Ventas
- **Facturas** con numeración automática
- **Control de pagos** en múltiples monedas (CUP, USD, EUR, MLC)
- **Descuentos** y promociones
- **Generación de recibos** en PDF

### 👥 Gestión de Clientes
- **Registro completo** de datos personales
- **Historial de compras**
- **Recetas médicas** con graduaciones
- **Notificaciones** para revisiones periódicas

### 🔧 Módulo de Taller
- **Órdenes de trabajo** con seguimiento
- **Asignación de tareas** a técnicos
- **Control de calidad**
- **Gestión de recursos** y mantenimiento

### 📊 Sistema de Reportes
- **Dashboard** con estadísticas en tiempo real
- **Reportes en PDF y Excel**
- **Gráficos interactivos**
- **Estadísticas de ventas, inventario y taller**

## 🚀 Instalación y Configuración

### Requisitos
- Python 3.8+
- Django 5.2+
- SQLite (incluido) o PostgreSQL
- Pillow para manejo de imágenes
- ReportLab para PDFs

### Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd optica_mipyme
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install django djangorestframework pillow reportlab openpyxl django-extensions python-decouple
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

6. **Ejecutar servidor**
```bash
python manage.py runserver
```

## 👤 Usuario de Prueba

**Administrador:**
- Usuario: `admin`
- Contraseña: `admin123`
- Email: `admin@optica.com`

## 🌐 Acceso al Sistema

Una vez iniciado el servidor, accede a:
- **Aplicación principal**: http://localhost:8000
- **Panel de administración**: http://localhost:8000/admin

## 📱 Interfaz de Usuario

### Dashboard Principal
- **Estadísticas en tiempo real**
- **Gráficos de ventas por mes**
- **Estado de órdenes del taller**
- **Productos con bajo stock**
- **Top vendedores del mes**

### Módulos Principales

#### 🛍️ Ventas
- Crear nuevas ventas
- Seleccionar cliente y productos
- Aplicar descuentos
- Registrar pagos en múltiples monedas
- Generar facturas PDF

#### 👥 Clientes
- Registro de nuevos clientes
- Gestión de recetas médicas
- Historial de compras
- Notificaciones automáticas

#### 📦 Productos
- **Monturas**: Código, nombre, material, género, precio, stock
- **Cristales**: Tipo, graduación, tratamientos, rangos de dioptrías
- Control de inventario automático

#### 🔧 Taller
- Órdenes de trabajo automáticas desde ventas
- Asignación de técnicos
- Seguimiento de tareas
- Control de calidad
- Gestión de recursos

## 🔧 Configuración Avanzada

### Variables de Entorno (.env)
```env
SECRET_KEY=tu-clave-secreta-aqui
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3
```

### Monedas Soportadas
- **CUP**: Pesos Cubanos
- **USD**: Dólares Americanos  
- **EUR**: Euros
- **MLC**: Moneda Libremente Convertible

### Roles de Usuario
- **administrador**: Acceso completo
- **vendedor**: Ventas y clientes
- **tecnico**: Taller y producción

## 📊 Funcionalidades Especiales

### Reportes Automáticos
- **Ventas diarias, semanales y mensuales**
- **Control de inventario**
- **Productividad del taller**
- **Clientes recurrentes**

### Notificaciones
- **Productos con bajo stock**
- **Órdenes atrasadas**
- **Revisiones de recetas vencidas**
- **Mantenimiento de equipos**

### API REST
- Endpoints para integración con otros sistemas
- Búsqueda de productos y clientes
- Estadísticas en JSON

## 🔄 Flujo de Trabajo Típico

1. **Administrador** registra productos (monturas y cristales)
2. **Vendedor** registra cliente y crea venta con receta
3. Sistema genera **orden de trabajo** automáticamente
4. **Técnico** procesa la orden en el taller
5. **Control de calidad** aprueba el trabajo
6. Cliente recibe notificación para **recoger producto**
7. Se registra **pago** y entrega
8. Sistema genera **reportes** automáticos

## 🛠️ Estructura del Proyecto

```
optica_mipyme/
├── usuarios/          # Gestión de usuarios y roles
├── productos/         # Monturas, cristales e inventario
├── clientes/          # Clientes y recetas médicas
├── ventas/           # Ventas, facturas y pagos
├── taller/           # Órdenes de trabajo y control de calidad
├── reportes/         # Dashboard y reportes
├── templates/        # Plantillas HTML
├── static/          # Archivos CSS, JS e imágenes
└── media/           # Uploads (fotos de productos)
```

## 🔧 Personalización

### Agregar Nuevos Campos
1. Modificar modelos en `models.py`
2. Crear migración: `python manage.py makemigrations`
3. Aplicar migración: `python manage.py migrate`

### Nuevos Reportes
1. Crear vista en `reportes/views.py`
2. Agregar URL en `reportes/urls.py`
3. Crear template correspondiente

### Integración con Hardware
- **Lectores de código de barras**
- **Impresoras térmicas**
- **Cajones de dinero**
- **Balanzas digitales**

## 🔒 Seguridad

- **Autenticación obligatoria** para todas las vistas
- **Control de permisos** por rol de usuario
- **Validación de datos** en formularios
- **Protección CSRF** habilitada
- **Sanitización** de inputs

## 📱 Funcionalidad Offline (Próximamente)

- **Cache local** de datos críticos
- **Sincronización automática** cuando hay conexión
- **Modo offline** para ventas básicas
- **Backup automático** de datos

## 🐛 Resolución de Problemas

### Error de Migración
```bash
python manage.py migrate --fake-initial
```

### Resetear Base de Datos
```bash
rm db.sqlite3
python manage.py migrate
python manage.py createsuperuser
```

### Problemas con Archivos Estáticos
```bash
python manage.py collectstatic
```

## 📝 Licencia

Este proyecto está desarrollado para uso educativo y comercial. 

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📞 Soporte

Para soporte técnico o consultas:
- **Email**: soporte@optica-mipyme.cu
- **Teléfono**: +53 xxxx-xxxx

---

**© 2024 Óptica MIPYME - Sistema de Gestión Integral**

*Desarrollado con ❤️ para las pequeñas empresas cubanas* 