# === app.py ===
from pathlib import Path

from flask import Flask

from app.containers import Application
from app.views import knowledge

# === Flask setup ===
root = Flask(__name__)


def create_app():
    container = Application()
    container.init_resources()
    container.wire(modules=[".views"])

    app = Flask(__name__)
    app.template_folder = Path(__file__).parent / "templates" / "web"
    app.register_blueprint(knowledge.app, url_prefix="/")

    return app


def main():
    create_app().run(debug=True)


# === Run Server ===
if __name__ == "__main__":
    main()
