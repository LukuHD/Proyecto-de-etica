"""
main.py — Punto de entrada principal de la aplicación FastAPI.

Este módulo es el núcleo del servidor backend local. Su responsabilidad es:
  1. Inicializar la aplicación FastAPI con metadatos descriptivos.
  2. Registrar los eventos de ciclo de vida (startup/shutdown) para garantizar
     que la base de datos JSON esté lista antes de atender cualquier petición.
  3. Montar los routers de los distintos controladores de rutas.
  4. Configurar el middleware CORS para permitir peticiones desde la extensión
     del navegador (que opera en un origen diferente, chrome-extension://).
  5. Exponer un endpoint de salud (/health) para que la extensión pueda
     verificar rápidamente que el backend local está activo.

Decisión arquitectónica: Se utiliza el patrón "lifespan" introducido en
FastAPI 0.93+ (que reemplaza los eventos on_event) para gestionar los
recursos de forma segura y expresiva mediante async context managers.
"""

import logging                    # Módulo estándar de registro de eventos
from contextlib import asynccontextmanager  # Para definir el ciclo de vida asíncrono

import uvicorn                    # Servidor ASGI de alto rendimiento para FastAPI
from fastapi import FastAPI        # Framework web principal
from fastapi.middleware.cors import CORSMiddleware  # Middleware para cabeceras CORS

# ── Importaciones internas ──────────────────────────────────────────────────
# Importamos el gestor de la base de datos JSON para inicializarla en el startup
from app.utils.db_manager import initialize_database

# Importamos el router de análisis donde viven los endpoints del motor IA
from app.routers.analysis import router as analysis_router

# ── Configuración del logger ────────────────────────────────────────────────
# Usamos el logger estándar de Python para que los mensajes de la aplicación
# sean diferenciables de los logs del framework y del servidor ASGI.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("desinformacion_shield")


# ── Ciclo de vida de la aplicación ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor del ciclo de vida completo de la aplicación.

    Todo el código ANTES del `yield` se ejecuta durante el STARTUP:
      - Inicializa el archivo JSON que actúa como base de datos local.
        Si el archivo ya existe, lo valida; si no, lo crea con un esquema vacío.

    Todo el código DESPUÉS del `yield` se ejecuta durante el SHUTDOWN:
      - En este caso, solo registramos el cierre limpio; no hay recursos
        de red abiertos (sockets de BD, pools de conexiones, etc.) que limpiar,
        ya que toda la persistencia es en archivo local.

    Usar asynccontextmanager en lugar de on_event es la práctica recomendada
    desde FastAPI ≥ 0.93 porque garantiza la liberación de recursos incluso
    si se produce una excepción durante el startup.
    """
    logger.info("🚀 Iniciando el motor de detección de desinformación…")

    # Inicializar (o validar) el archivo JSON de base de datos local.
    # Esta llamada es await-able porque, en implementaciones avanzadas,
    # la escritura inicial puede ser asíncrona (usando aiofiles).
    await initialize_database()

    logger.info("✅ Base de datos JSON inicializada y lista.")
    logger.info("🧠 Motor de IA listo para recibir publicaciones.")

    yield  # ← La aplicación está activa y atendiendo peticiones

    # ── Código de SHUTDOWN ──────────────────────────────────────────────────
    logger.info("🛑 Cerrando el servidor de análisis de forma segura…")


# ── Instancia de la aplicación FastAPI ─────────────────────────────────────
app = FastAPI(
    title="Desinformación Shield — Motor Analítico Local",
    description=(
        "Backend local de alto rendimiento que analiza publicaciones de redes "
        "sociales usando IA multimodal (texto + imagen) para detectar "
        "desinformación, fraudes y noticias falsas en tiempo real."
    ),
    version="1.0.0",
    docs_url="/docs",        # Swagger UI disponible solo en desarrollo local
    redoc_url="/redoc",      # Redoc UI alternativa
    lifespan=lifespan,       # Registrar el gestor de ciclo de vida
)

# ── Middleware CORS ─────────────────────────────────────────────────────────
# Las extensiones de Chrome/Firefox se ejecutan bajo el origen especial
# "chrome-extension://<ID>" o "moz-extension://<ID>". El middleware CORS
# debe permitir estos orígenes para que los fetch() de la extensión funcionen.
# En producción local se puede restringir a los IDs de extensión específicos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Para desarrollo; restringir en despliegue
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Registro de routers ─────────────────────────────────────────────────────
# Montamos el router de análisis bajo el prefijo /api/v1 para versionar la API
# y facilitar migraciones futuras sin romper la compatibilidad con la extensión.
app.include_router(
    analysis_router,
    prefix="/api/v1",
    tags=["Análisis de Publicaciones"],
)


# ── Endpoint de salud ───────────────────────────────────────────────────────
@app.get("/health", tags=["Sistema"])
async def health_check():
    """
    Endpoint de verificación de estado (health check).

    La extensión del navegador puede consultar este endpoint en intervalos
    regulares (polling ligero) para confirmar que el backend local está activo
    antes de enviar una publicación para análisis. Si recibe 200 OK con
    {"status": "ok"}, continúa normalmente; si el servidor está caído,
    puede mostrar un ícono de advertencia en la barra de herramientas.
    """
    return {"status": "ok", "service": "Desinformación Shield", "version": "1.0.0"}


# ── Punto de entrada directo ────────────────────────────────────────────────
# Permite ejecutar el servidor directamente con `python -m app.main`
# sin necesidad de invocar uvicorn desde la línea de comandos.
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",          # Ruta al objeto FastAPI dentro del módulo
        host="127.0.0.1",        # Solo escucha en loopback — no expone en red local
        port=8000,               # Puerto estándar para el backend local
        reload=False,            # Desactivar recarga automática en producción
        workers=1,               # Un solo worker — el modelo IA ocupa toda la RAM GPU
        log_level="info",
    )
