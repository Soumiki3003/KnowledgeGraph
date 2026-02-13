# === ROUTE: File Upload + Dual Parser ===
from dependency_injector.wiring import Provide, inject
from flask import Blueprint, render_template, request
from flask_pydantic import validate

from app import controllers, schemas
from app.containers import Application

app = Blueprint("knowledge", __name__)


@app.route("/upload", methods=["GET", "POST"])
@validate()
@inject
def upload(
    form: schemas.KnowledgeUploadRequest,
    *,
    knowledge_controller: controllers.KnowledgeController = Provide[
        Application.controllers.knowledge_controller
    ],
    allowed_extensions: list[str] = Provide[Application.config.core.allowed_extensions],
) -> str | schemas.KnowledgeUploadResponse:
    # TODO: test this generation
    if request.method == "POST":
        return knowledge_controller.parse_uploaded_file_list(form)
    return render_template("upload.html", allowed_extensions=allowed_extensions)
