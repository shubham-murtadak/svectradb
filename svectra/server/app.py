"""
FastAPI application factory and middleware configuration for SvectraDB.
"""

from contextlib import asynccontextmanager
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from svectra.core.db import SvectraDB
from svectra.core.models import SystemMetrics


def get_db() -> SvectraDB:
    """Return database instance, initializing lazily if needed."""
    if not hasattr(app.state, "db") or app.state.db is None:
        persist_dir = os.environ.get("SVECTRA_DATA_DIR", "./svectra_data")
        app.state.db = SvectraDB(persist_dir=persist_dir)
        app.state.start_time = time.time()
        app.state.total_queries = 0
        app.state.total_query_latency_ms = 0.0
    return app.state.db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure DB is initialized
    get_db()
    yield
    # Shutdown: Flush all collections and close files
    if hasattr(app.state, "db") and app.state.db:
        app.state.db.flush_all()
        app.state.db.close()


app = FastAPI(
    title="SvectraDB API",
    description="Ultra-fast, production-grade vector database engineered for AI applications.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for all frontends & microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
from svectra.server.routes import router
app.include_router(router)

# Serve Embedded Static Dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def serve_dashboard():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>SvectraDB Server is Running</h1><p>Visit <a href='/docs'>/docs</a> for API</p>")
