"""
Flask application factory.

Why an app factory?
-------------------
- Keeps global state minimal.
- Enables clean modular route registration via Blueprints.
- Makes testing easier (create_app(test_config=...)).
"""

from __future__ import annotations

from pathlib import Path
from flask import Flask
from dotenv import load_dotenv

# from ui.routes import register_blueprints
from ..routes import register_blueprints

def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns
    -------
    Flask
        Configured Flask app instance.2dq 
    """
    print(">>> create_app started", flush=True)
    # Load .env once at startup (repo root)
    REPO_ROOT = Path(__file__).resolve().parents[2]
    # parents[2] because: ui/ui_app/factory.py → go up to ui/ (1) → repo root (2)
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(">>> dotenv loaded", flush=True)

    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    # Dev: don't let the browser cache our static JS/CSS. Without this, edits to
    # files like the chunk-scores panel won't appear on a normal refresh (the
    # browser serves a stale cached copy), which looks like "no changes".
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    print(">>> Flask created", flush=True)

    # Dev: force the browser to NEVER cache static assets. `no-cache` only asks
    # the browser to revalidate, and some browsers still serve ES modules from
    # the in-memory cache without a network round-trip on a normal refresh —
    # which makes JS/CSS edits look like they "didn't apply". `no-store` removes
    # that ambiguity so edits always show up after a reload.
    @app.after_request
    def _no_store_static(response):  # type: ignore[unused-variable]
        from flask import request

        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers.pop("Expires", None)
            response.headers.pop("ETag", None)
        return response

    # Register modular routes (Blueprints)
    register_blueprints(app)
    print(">>> blueprints registered", flush=True)

    return app