# === app.py ===
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from app.containers import Application
from app.views import knowledge


# TODO: test upload route
def create_app():
    container = Application()
    container.init_resources()
    container.wire(modules=["app.views.knowledge"])

    app = Flask(__name__)
    app.template_folder = Path(__file__).parent / "app" / "templates" / "web"
    app.register_blueprint(knowledge.app, url_prefix="/")

    return app


def main():
    load_dotenv()
    create_app().run(debug=True)


# === Run Server ===
if __name__ == "__main__":
    main()
