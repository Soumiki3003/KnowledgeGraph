
from dependency_injector.wiring import Provide, inject
from flask import Blueprint, render_template, request
from flask_pydantic import validate

from app import controllers, schemas
from app.containers import Application

app = Blueprint("knowledge", __name__)


@app.route("/upload", methods=["GET"])
@inject
def upload(
    *,
    allowed_extensions: list[str] = Provide[Application.core.allowed_extensions],
):
    return render_template(
        "knowledge/upload.html",
        allowed_extensions=allowed_extensions,
    )


@app.route("/upload/list", methods=["GET"])
@inject
def upload_list(
    *,
    knowledge_controller: controllers.KnowledgeController = Provide[
        Application.controllers.knowledge_controller
    ],
):
    page = request.args.get("page", 1, type=int)
    page_size = request.args.get("page_size", 10, type=int)

    uploads = knowledge_controller.get_uploads(page=page, page_size=page_size)

    return render_template(
        "knowledge/upload_list.html",
        uploads=uploads,
    )


@app.route("/upload/submit", methods=["POST"])
@validate()
@inject
def upload_submit(
    *,
    knowledge_controller: controllers.KnowledgeController = Provide[
        Application.controllers.knowledge_controller
    ],
):
    form = schemas.KnowledgeUploadRequest(
        files=request.files.getlist("files"),
        html_link=request.form.get("html_link"),
    )
    response = knowledge_controller.parse_uploaded_file_list(form)
    return response
