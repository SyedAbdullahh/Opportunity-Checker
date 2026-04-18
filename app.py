import base64
import os
from datetime import datetime
from email.utils import parsedate_to_datetime

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

load_dotenv()

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]


app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET", "change-me-in-env")

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:5000/auth/google/callback")


def _require_config() -> None:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("CLIENT_ID and CLIENT_SECRET must be set in .env")


def _build_flow() -> Flow:
    _require_config()
    client_config = {
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }
    }
    return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)


def _decode_body(payload: dict) -> str:
    body = payload.get("body", {})
    data = body.get("data")
    if data:
        return _decode_base64(data)

    for part in payload.get("parts", []):
        mime_type = part.get("mimeType", "")
        part_data = part.get("body", {}).get("data")
        if part_data and mime_type in {"text/plain", "text/html"}:
            return _decode_base64(part_data)

    return ""


def _decode_base64(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_header(headers: list[dict], key: str) -> str:
    key_lower = key.lower()
    for h in headers:
        if h.get("name", "").lower() == key_lower:
            return h.get("value", "")
    return ""


def _parse_internal_date(internal_date_ms: str) -> str:
    try:
        dt = datetime.utcfromtimestamp(int(internal_date_ms) / 1000)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ""


def _parse_date_header(raw_date: str) -> str:
    if not raw_date:
        return ""
    try:
        dt = parsedate_to_datetime(raw_date)
        return dt.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return raw_date


def _get_gmail_service():
    creds_data = session.get("google_credentials")
    if not creds_data:
        return None

    from google.oauth2.credentials import Credentials

    creds = Credentials(
        token=creds_data.get("token"),
        refresh_token=creds_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPES,
    )
    return build("gmail", "v1", credentials=creds)


@app.route("/")
def index():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login")
def login():
    return render_template("login.html", user=session.get("user"))


@app.route("/auth/google/start")
def auth_google_start():
    flow = _build_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    session["oauth_state"] = state
    return redirect(auth_url)


@app.route("/auth/google/callback")
def auth_google_callback():
    state = session.get("oauth_state")
    flow = _build_flow()
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    session["google_credentials"] = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
    }

    oauth2_service = build("oauth2", "v2", credentials=credentials)
    info = oauth2_service.userinfo().get().execute()
    session["user"] = {
        "name": info.get("name") or info.get("email"),
        "email": info.get("email"),
        "picture": info.get("picture"),
    }

    if state != request.args.get("state"):
        session.clear()
        return redirect(url_for("login"))

    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    user = session.get("user")
    if not user:
        return redirect(url_for("login"))

    return render_template("dashboard.html", user=user, emails=None, error=None, scanned_count=0)


@app.route("/scan", methods=["POST"])
def scan():
    user = session.get("user")
    if not user:
        return redirect(url_for("login"))

    max_results = request.form.get("max_results", "10")
    try:
        max_results_int = max(1, min(15, int(max_results)))
    except ValueError:
        max_results_int = 10

    gmail = _get_gmail_service()
    if gmail is None:
        return render_template(
            "dashboard.html",
            user=user,
            emails=None,
            error="Session expired. Please sign in again.",
            scanned_count=0,
        )

    try:
        query = (
            "subject:(scholarship OR internship OR fellowship OR competition OR admission OR grant OR program) "
            "newer_than:365d"
        )
        messages_result = gmail.users().messages().list(
            userId="me", q=query, maxResults=max_results_int
        ).execute()
        messages = messages_result.get("messages", [])

        parsed_emails = []
        for msg in messages:
            detail = gmail.users().messages().get(
                userId="me",
                id=msg["id"],
                format="full",
            ).execute()

            payload = detail.get("payload", {})
            headers = payload.get("headers", [])
            subject = _extract_header(headers, "Subject")
            sender = _extract_header(headers, "From")
            date_header = _extract_header(headers, "Date")
            body_text = _decode_body(payload)
            snippet = detail.get("snippet", "")

            preview = body_text.strip().replace("\n", " ").replace("\r", " ")
            preview = " ".join(preview.split())
            if not preview:
                preview = snippet
            preview = (preview[:350] + "...") if len(preview) > 350 else preview

            parsed_emails.append(
                {
                    "subject": subject or "(No Subject)",
                    "from": sender or "(Unknown Sender)",
                    "date": _parse_date_header(date_header)
                    or _parse_internal_date(detail.get("internalDate", "")),
                    "snippet": snippet,
                    "preview": preview,
                }
            )

        return render_template(
            "dashboard.html",
            user=user,
            emails=parsed_emails,
            error=None,
            scanned_count=len(parsed_emails),
        )
    except Exception as ex:
        return render_template(
            "dashboard.html",
            user=user,
            emails=None,
            error=f"Failed to fetch emails: {ex}",
            scanned_count=0,
        )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
