import os
from datetime import datetime

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from pymongo.errors import PyMongoError

from db import analysis_collection, users_collection
from fusion.prediction_pipeline import run_multimodal_pipeline


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY", "baby-cry-analyzer-development-key"
)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def current_user():
    return session.get("user")


@app.context_processor
def inject_user():
    return {"user": current_user()}


@app.route("/")
def home():
    if current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template(
                "login.html", error="Please enter your email and password."
            )

        try:
            user = users_collection.find_one({"email": email})
        except PyMongoError:
            return render_template(
                "login.html", error="Database unavailable. Please try again later."
            )
        if not user or not check_password_hash(user["password"], password):
            return render_template(
                "login.html", error="Invalid email or password."
            )

        session["user"] = {"name": user["username"], "email": user["email"]}
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or len(password) < 8:
            return render_template(
                "signup.html",
                error="Complete every field and use at least 8 password characters.",
            )

        try:
            if users_collection.find_one({"email": email}):
                return render_template(
                    "signup.html",
                    error="An account with this email already exists.",
                )

            hashed = generate_password_hash(password)
            users_collection.insert_one({
                "username": username,
                "email": email,
                "password": hashed,
                "created_at": datetime.utcnow(),
            })
        except PyMongoError:
            return render_template(
                "signup.html",
                error="Database unavailable. Please try again later.",
            )

        session["user"] = {"name": username, "email": email}
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not current_user():
        return redirect(url_for("login"))
    return render_template("homepage.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if not current_user():
        return redirect(url_for("login"))

    video = request.files.get("video")
    if not video or not video.filename:
        return render_template(
            "homepage.html", upload_error="Please choose a video to analyze."
        )

    if not allowed_file(video.filename):
        return render_template(
            "homepage.html",
            upload_error="Use an MP4, MOV, AVI, MKV, or WebM video.",
        )

    filename = secure_filename(video.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stored_filename = f"{timestamp}_{filename}"
    video_path = os.path.join(UPLOAD_FOLDER, stored_filename)
    video.save(video_path)

    result = run_multimodal_pipeline(video_path)
    if result.get("error"):
        return render_template(
            "homepage.html",
            upload_error=f"Analysis could not be completed: {result['error']}",
        )

    user = current_user()
    try:
        serializable_result = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in result.items()
        }
        analysis_collection.insert_one({
            "username": user["name"],
            "email": user["email"],
            "timestamp": datetime.now(),
            "original_name": filename,
            "video_file": stored_filename,
            "result": serializable_result,
        })
    except PyMongoError:
        return render_template(
            "homepage.html",
            upload_error="Could not save the result. Please try again.",
        )

    session["last_result"] = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in result.items()
    }
    session["last_video"] = stored_filename
    session["last_original_name"] = filename

    return render_template(
        "result.html",
        result=result,
        video_file=stored_filename,
        original_name=filename,
    )


@app.route("/results")
def results():
    if not current_user():
        return redirect(url_for("login"))

    result = session.get("last_result")
    video_file = session.get("last_video")
    if not result or not video_file:
        return render_template("result.html", result=None)

    return render_template(
        "result.html",
        result=result,
        video_file=video_file,
        original_name=session.get("last_original_name", video_file),
    )


@app.route("/history")
def history():
    if not current_user():
        return redirect(url_for("login"))

    user = current_user()
    try:
        items = list(
            analysis_collection.find(
                {"email": user["email"]},
                sort=[("timestamp", -1)],
            )
        )
    except PyMongoError:
        items = []

    return render_template("history.html", history=items)


@app.route("/history/<int:item_index>")
def history_item(item_index):
    if not current_user():
        return redirect(url_for("login"))

    user = current_user()
    try:
        items = list(
            analysis_collection.find(
                {"email": user["email"]},
                sort=[("timestamp", -1)],
            )
        )
    except PyMongoError:
        items = []

    if item_index < 0 or item_index >= len(items):
        return redirect(url_for("history"))

    item = items[item_index]
    serializable_result = {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in item["result"].items()
    }
    return render_template(
        "result.html",
        result=serializable_result,
        video_file=item["video_file"],
        original_name=item["original_name"],
    )


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)



# .\.venv\Scripts\Activate.ps1                                                    
# >> python app.py                                                         