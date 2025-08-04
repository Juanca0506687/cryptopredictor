# 🔍 Sistema de Gestión para Óptica MIPYME

## 📋 Descripción

Sistema integral de gestión desarrollado específicamente para MIPYMES cubanas del sector óptico. Permite gestionar ventas de espejuelos graduados, monturas, cristales y servicios relacionados, con capacidad de funcionamiento offline y sincronización en red local.

## ✨ Características Principales

### 🛠️ Módulos Implementados

#### 1️⃣ **Gestión de Usuarios**
- Sistema multiusuario con roles específicos
- **Administrador**: Acceso completo al sistema
- **Vendedor**: Gestión de ventas y clientes
- **Técnico**: Manejo del taller y producción
- Control de sesiones y auditoría
- Perfiles de usuario personalizables

#### 2️⃣ **Gestión de Productos**
- Registro completo de monturas con especificaciones técnicas
- Catálogo de cristales con diferentes graduaciones
- Control de inventario en tiempo real
- Alertas automáticas de stock bajo
- Gestión de marcas y categorías
- Movimientos de inventario detallados

#### 3️⃣ **Gestión de Ventas**
- Creación de facturas profesionales
- Soporte para múltiples monedas (CUP, USD, EUR, MLC)
- Sistema de descuentos flexible
- Control de pagos parciales y completos
- Gestión de devoluciones
- Tasas de cambio actualizables

#### 4️⃣ **Clientes y Recetas**
- Registro completo de datos del cliente
- Almacenamiento seguro de recetas oftálmicas
- Historial de compras y servicios
- Sistema de notificaciones para revisiones
- Validación de datos según estándares cubanos

#### 5️⃣ **Taller de Producción**
- Órdenes de trabajo automatizadas
- Asignación de tareas a técnicos
- Control de tiempos y productividad
- Seguimiento de estado de pedidos
- Control de calidad integrado
- Gestión de herramientas y equipos

#### 6️⃣ **Sistema de Reportes**
- Reportes de ventas (diarios, semanales, mensuales)
- Control de inventario
- Análisis de clientes recurrentes
- Estadísticas de productividad del taller
- Exportación a PDF y Excel

## 🚀 Instalación y Configuración

### Requisitos del Sistema
- Python 3.8 o superior
- Django 4.2.7
- SQLite (incluido) o PostgreSQL/MySQL (opcional)
- 2GB RAM mínimo
- 1GB espacio en disco

### Instalación Rápida

```bash
# 1. Clonar o descargar el proyecto
cd optica_mipyme

# 2. Instalar dependencias
pip install -r requirements_django.txt

# 3. Realizar migraciones
python manage.py migrate

# 4. Crear superusuario (ya creado: admin/admin123)
python manage.py createsuperuser

# 5. Ejecutar servidor
python manage.py runserver
```

### Acceso al Sistema
- **URL**: http://localhost:8000
- **Usuario**: admin
- **Contraseña**: admin123
- **Panel Admin**: http://localhost:8000/admin/

## 👥 Usuarios del Sistema

### Credenciales Predeterminadas
- **Administrador**: admin / admin123

### Creación de Usuarios
1. Acceder como administrador
2. Ir a "Gestión de Usuarios"
3. Hacer clic en "Nuevo Usuario"
4. Completar formulario con rol apropiado

## 🎯 Flujo de Trabajo Típico

### Para Administradores
1. **Configuración inicial**:
   - Crear usuarios (vendedores, técnicos)
   - Configurar categorías y marcas
   - Establecer tasas de cambio
   - Registrar productos iniciales

2. **Operación diaria**:
   - Supervisar ventas y taller
   - Revisar reportes
   - Gestionar inventario
   - Administrar usuarios

### Para Vendedores
1. **Atención al cliente**:
   - Registrar nuevo cliente
   - Crear receta oftálmica
   - Seleccionar productos
   - Generar factura

2. **Seguimiento**:
   - Verificar estado de órdenes
   - Gestionar pagos
   - Coordinar entregas
   - Atender devoluciones

### Para Técnicos
1. **Producción**:
   - Revisar órdenes asignadas
   - Iniciar trabajos
   - Registrar progreso
   - Completar órdenes

2. **Control de calidad**:
   - Inspeccionar productos
   - Registrar defectos
   - Aprobar entregas

## 💰 Gestión de Monedas

El sistema soporta las monedas utilizadas en Cuba:
- **CUP**: Peso Cubano (moneda base)
- **USD**: Dólar Estadounidense
- **EUR**: Euro
- **MLC**: Moneda Libremente Convertible

### Configuración de Tasas
Las tasas de cambio se configuran desde el panel de administración y se aplican automáticamente en las transacciones.

## 📊 Reportes Disponibles

### Reportes de Ventas
- Ventas diarias, semanales, mensuales
- Análisis por vendedor
- Productos más vendidos
- Ingresos por moneda

### Reportes de Inventario
- Stock actual por producto
- Productos con stock bajo
- Movimientos de inventario
- Valoración de inventario

### Reportes de Taller
- Productividad por técnico
- Órdenes pendientes y completadas
- Tiempos de producción
- Control de calidad

### Reportes de Clientes
- Clientes más frecuentes
- Historial de compras
- Clientes que necesitan revisión
- Análisis demográfico

## 🔧 Configuración Avanzada

### Base de Datos
Por defecto usa SQLite. Para producción, se recomienda PostgreSQL:

```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'optica_db',
        'USER': 'optica_user',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### Backup Automático
```bash
# Crear backup
python manage.py dumpdata > backup_$(date +%Y%m%d).json

# Restaurar backup
python manage.py loaddata backup_20241201.json
```

### Configuración de Red Local
Para acceso en red local, modificar `ALLOWED_HOSTS` en `settings.py`:
```python
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '192.168.1.*']
```

## 🛡️ Seguridad

### Medidas Implementadas
- Autenticación basada en roles
- Validación de permisos por vista
- Registro de actividades
- Sesiones seguras
- Validación de formularios
- Protección CSRF

### Recomendaciones
- Cambiar contraseñas predeterminadas
- Realizar backups regulares
- Actualizar el sistema periódicamente
- Usar HTTPS en producción
- Restringir acceso físico al servidor

## 📱 Compatibilidad

### Navegadores Soportados
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

### Dispositivos
- **Desktop**: Experiencia completa
- **Tablet**: Interfaz adaptativa
- **Móvil**: Funciones básicas

## 🆘 Solución de Problemas

### Problemas Comunes

#### Error de Migración
```bash
python manage.py migrate --fake-initial
```

#### Puerto en Uso
```bash
python manage.py runserver 8001
```

#### Problemas de Permisos
```bash
chmod +x manage.py
```

### Logs del Sistema
Los logs se almacenan en:
- Django: consola del servidor
- Base de datos: tabla de auditoría
- Errores: archivo de log (si configurado)

## 🔄 Actualizaciones

### Proceso de Actualización
1. Hacer backup de la base de datos
2. Descargar nueva versión
3. Instalar dependencias actualizadas
4. Ejecutar migraciones
5. Reiniciar servidor

## 📞 Soporte

### Recursos de Ayuda
- **Manual de Usuario**: Incluido en el sistema
- **Panel de Administración**: Gestión avanzada
- **Logs del Sistema**: Diagnóstico de problemas

### Información del Desarrollador
- **Versión**: 1.0.0
- **Desarrollado para**: MIPYMES cubanas
- **Tecnología**: Django + Bootstrap 5
- **Licencia**: Uso comercial permitido

## 🎨 Personalización

### Temas y Colores
El sistema usa CSS personalizable en `templates/base.html`. Los colores principales se definen en variables CSS:

```css
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --accent-color: #e74c3c;
}
```

### Logotipo
Para cambiar el logotipo, reemplazar el ícono en la barra de navegación.

## 📈 Escalabilidad

### Para Múltiples Sucursales
- Configurar base de datos centralizada
- Implementar sincronización de datos
- Gestión de inventario por ubicación
- Reportes consolidados

### Integración con Otros Sistemas
- API REST (desarrollo futuro)
- Exportación de datos
- Importación desde otros sistemas
- Sincronización con contabilidad

## ✅ Estado del Proyecto

### Módulos Completados ✅
- [x] Sistema de usuarios y autenticación
- [x] Gestión de productos e inventario
- [x] Módulo de ventas y facturación
- [x] Gestión de clientes y recetas
- [x] Taller y órdenes de trabajo
- [x] Interfaz de usuario moderna
- [x] Panel de administración

### Funcionalidades Pendientes 🔄
- [ ] Generación de reportes PDF
- [ ] Exportación a Excel
- [ ] Sistema de notificaciones
- [ ] API REST
- [ ] Aplicación móvil
- [ ] Sincronización multi-sucursal

---

## 🎉 ¡Listo para Usar!

El sistema está completamente funcional y listo para ser usado en una MIPYME de óptica. Incluye todas las funcionalidades básicas necesarias para la gestión integral del negocio.

**¡Bienvenido a la era digital de tu óptica! 🔍👓**