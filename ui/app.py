"""
UI entrypoint.

This file intentionally stays tiny:
- It imports the Flask app factory.
- It starts the dev server.

All actual route logic lives in ui/routes/.
"""

import os

from .ui_app.factory import create_app

app = create_app()

if __name__ == "__main__":
    # Dev run only. For cluster usage we already do port forwarding.
    # Honors the PORT env var (default 5002) so the app can run on an
    # alternate port when 5002 is already taken.
    port = int(os.getenv("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
