from flask import Blueprint, abort, render_template, request
from flask_login import current_user, login_required

from app.surveys import SURVEY_LIST, SURVEYS, submit_survey

app = Blueprint("survey", __name__)


@app.route("/survey/list")
@login_required
def list_surveys():
    return render_template("survey/list.html", surveys=SURVEY_LIST)


@app.route("/survey/<survey_id>")
@login_required
def show_form(survey_id):
    survey = SURVEYS.get(survey_id)
    if not survey:
        abort(404)
    return render_template("survey/form.html", survey=survey)


@app.route("/survey/<survey_id>/submit", methods=["POST"])
@login_required
def submit(survey_id):
    survey = SURVEYS.get(survey_id)
    if not survey:
        abort(404)

    try:
        submit_survey(
            survey=survey,
            form_data=request.form,
            student_name=current_user.name,
            student_email=current_user.email,
        )
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)

    return render_template(
        "survey/success.html",
        survey=survey,
        success=success,
        error=error,
    )
