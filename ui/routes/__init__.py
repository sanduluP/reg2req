"""
Blueprint registration.

This module exposes one function `register_blueprints(app)` that keeps
all blueprint wiring in one place.
"""

from __future__ import annotations

from flask import Flask, render_template

from .comparison_routes import comparison_bp
from .graph_routes import graph_bp
from .pipeline_routes import pipeline_bp
from .verification_routes import verification_bp

def register_blueprints(app: Flask) -> None:
    """
    Register all UI Blueprints on the Flask app.

    Parameters
    ----------
    app:
        Flask application instance.
    """
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.get("/health")
    def health():
        return {"status": "ok"}
    
    app.register_blueprint(graph_bp, url_prefix="/api/graph")
    app.register_blueprint(pipeline_bp, url_prefix="/api/pipeline")
    app.register_blueprint(comparison_bp, url_prefix="/api/comparison")
    app.register_blueprint(verification_bp, url_prefix="/api/verification")
