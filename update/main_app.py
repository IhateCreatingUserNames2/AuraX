# main_app.py
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Import the refactored routes and database setup
from api.routes import app as routes_app
from database.models import DatabaseSetup

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MainApp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifecycle Manager.
    Handles startup initialization and graceful shutdown.
    """
    logger.info("🏁 Starting CEAF V4 (Project Race Car)...")

    # 1. Initialize Async Database Connection (PostgreSQL)
    try:
        logger.info("🔌 Initializing Database Connection...")
        await DatabaseSetup.init_db()
        logger.info("✅ Database Tables Verified.")
    except Exception as e:
        logger.critical(f"❌ Database Initialization Failed: {e}")
        raise e

    # 2. (Optional) Check Redis/Qdrant connectivity here if strict startup required

    yield

    logger.info("🛑 Shutting down CEAF V4...")


# Initialize FastAPI
app = FastAPI(
    title="Aura Cognitive Architecture V4",
    description="Distributed Event-Driven Cognitive System",
    version="4.0.0",
    lifespan=lifespan
)

# Mount the API Routes
app.mount("/ceaf", routes_app)

# Mount Static Files (for frontend)
# Ensure the 'frontend' directory exists or remove this block
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    # Hot reload enabled for development
    uvicorn.run(
        "main_app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )