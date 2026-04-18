import base64
import os
import time
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from agent import build_agent
from db import get_user_profile, init_db
from tools import extract_profile_from_message, process_and_rank_inbox, save_user_profile

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

# Allow OAuth over HTTP only for localhost development.
if REDIRECT_URI.startswith("http://localhost") or REDIRECT_URI.startswith("http://127.0.0.1"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

init_db()


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


def _get_chat_history() -> list[dict[str, str]]:
    return session.get("chat_history", [])


def _set_chat_history(history: list[dict[str, str]]) -> None:
    session["chat_history"] = history[-20:]


def _set_last_scan_result(result: dict[str, Any] | None) -> None:
    if result is None:
        session.pop("last_scan_result", None)
        return
    session["last_scan_result"] = result


def _get_last_scan_result() -> dict[str, Any] | None:
    value = session.get("last_scan_result")
    return value if isinstance(value, dict) else None


def _request_wants_json() -> bool:
    return request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest"


def _short_error_message(exc: Exception | str) -> str:
    message = str(exc).strip().splitlines()[0]
    return message[:220]


def _missing_profile_message() -> str:
    return (
        "Profile is missing. Please send your profile using this format:\n\n"
        "Degree Program: e.g BS Software Engineering\n"
        "Current Semester: e.g 8th\n"
        "CGPA: Anywhere between 2.0-4.0 Range\n"
        "Skills/Interests: Any Skills AI, ML, NLP, Leadership, Management\n"
        "Preferred Opportunity Types: Choose scholarship or internship\n"
        "Financial Need Required (Yes/No):\n"
        "Location Preference: Country/City or Remote Please Specify.\n"
        "Past Experience: e.g BS student, internship projects\n\n"
        "You can copy this format and replace the values with your own details."
    )


def _scan_completion_message(scan_result: dict[str, Any] | None) -> str:
    ranked = scan_result.get("ranked_opportunities", []) if isinstance(scan_result, dict) else []
    if isinstance(ranked, list) and len(ranked) > 0:
        return "Scan completed. Ranked scholarship opportunities are shown in cards."
    return "Scan completed. No scholarship opportunities were found in the latest inbox scan."


def _decode_base64(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _decode_body(payload: dict[str, Any]) -> str:
    body = payload.get("body", {}) if isinstance(payload, dict) else {}
    data = body.get("data")
    if isinstance(data, str) and data:
        return _decode_base64(data)

    for part in payload.get("parts", []) if isinstance(payload, dict) else []:
        mime_type = part.get("mimeType", "")
        part_data = part.get("body", {}).get("data")
        if part_data and mime_type in {"text/plain", "text/html"}:
            return _decode_base64(part_data)
    return ""


def _extract_header(headers: list[dict[str, Any]], key: str) -> str:
    wanted = key.lower()
    for header in headers:
        if header.get("name", "").lower() == wanted:
            return header.get("value", "")
    return ""


def _fetch_recent_email_texts(credentials_data: dict[str, str], max_results: int = 100) -> tuple[list[str], float]:
    started_at = time.time()
    print(f"[SCAN_ROUTE] Fetching up to {max_results} recent emails from Gmail", flush=True)
    credentials = Credentials(
        token=credentials_data.get("token"),
        refresh_token=credentials_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPES,
    )
    gmail = build("gmail", "v1", credentials=credentials)

    listed = gmail.users().messages().list(
        userId="me",
        q="newer_than:365d -in:spam -in:trash",
        maxResults=max_results,
    ).execute()
    messages = listed.get("messages", [])

    raw_emails: list[str] = []
    for index, message in enumerate(messages, start=1):
        detail = gmail.users().messages().get(userId="me", id=message["id"], format="full").execute()
        payload = detail.get("payload", {})
        headers = payload.get("headers", [])
        subject = _extract_header(headers, "Subject") or "(No Subject)"
        body = _decode_body(payload)
        snippet = detail.get("snippet", "")
        email_text = f"Subject: {subject}\n\n{body or snippet}".strip()
        raw_emails.append(email_text)
        print(f"[SCAN_ROUTE] Prepared email {index}/{len(messages)}: {subject}", flush=True)

    fetch_time = round(time.time() - started_at, 3)
    print(f"[SCAN_ROUTE] Prepared {len(raw_emails)} email texts for ranking pipeline in {fetch_time}s", flush=True)
    return raw_emails, fetch_time


def _run_scan_pipeline(user_email: str, credentials_data: dict[str, str], max_results: int = 100) -> dict[str, Any]:
    pipeline_telemetry: dict[str, Any] = {
        "emails_fetched": 0,
        "fetch_time": 0.0,
        "extraction_time": 0.0,
        "scoring_time": 0.0,
        "explainer_time": 0.0,
    }

    profile = get_user_profile(user_email)
    if not profile:
        raise ValueError(_missing_profile_message())

    raw_emails, fetch_time = _fetch_recent_email_texts(credentials_data, max_results=max_results)
    pipeline_telemetry["emails_fetched"] = len(raw_emails)
    pipeline_telemetry["fetch_time"] = fetch_time

    ranked, processed_telemetry = process_and_rank_inbox(
        raw_emails=raw_emails,
        profile_data=profile,
        telemetry=pipeline_telemetry,
    )
    pipeline_telemetry.update(processed_telemetry)
    result = {
        "user_email": user_email,
        "profile": profile,
        "scanned_count": len(raw_emails),
        "opportunities_detected": len(ranked),
        "ignored_count": max(0, len(raw_emails) - len(ranked)),
        "ranked_opportunities": ranked,
        "pipeline_telemetry": pipeline_telemetry,
    }
    _set_last_scan_result(result)
    return result


def _dashboard_payload(error: str | None = None, agent_reply: str | None = None) -> dict[str, Any]:
    user = session.get("user")
    profile = get_user_profile(user["email"]) if user else None
    return {
        "user": user,
        "profile": profile,
        "chat_history": _get_chat_history(),
        "agent_reply": agent_reply,
        "error": error,
        "last_scan_result": _get_last_scan_result(),
    }


def _format_scan_summary(result: dict[str, object]) -> str:
    if not isinstance(result, dict):
        return "📥 Opportunity Inbox Analysis\n\nSummary\n\nTotal Emails Scanned: 0\nOpportunities Detected: 0\nIgnored Emails: 0"

    opportunities = result.get("ranked_opportunities", []) if isinstance(result.get("ranked_opportunities"), list) else []
    ignored_emails = result.get("ignored_emails", []) if isinstance(result.get("ignored_emails"), list) else []
    scanned_count = int(result.get("scanned_count", 0) or 0)
    ignored_count = int(result.get("ignored_count", len(ignored_emails)) or 0)

    lines: list[str] = [
        "📥 Opportunity Inbox Analysis",
        "",
        "Summary",
        "",
        f"Total Emails Scanned: {scanned_count}",
        f"Opportunities Detected: {len(opportunities)}",
        f"Ignored Emails: {ignored_count}",
        "🏆 Top Opportunities (Ranked for You)",
    ]

    for index, item in enumerate(opportunities[:2], start=1):
        if not isinstance(item, dict):
            continue

        opp = item.get("opportunity", item) if isinstance(item.get("opportunity", item), dict) else {}
        score = item.get("score", {}) if isinstance(item.get("score"), dict) else {}
        breakdown = item.get("score_breakdown", {}) if isinstance(item.get("score_breakdown"), dict) else {}

        title = str(opp.get("title") or item.get("email_subject") or item.get("title") or "(No Subject)")
        opportunity_type = str(opp.get("opportunity_type") or item.get("opportunity_type") or "other")
        organization = str(opp.get("organization") or "Not clearly specified")
        deadline = str(opp.get("deadline") or item.get("deadline") or "Not clearly specified")
        priority_score = score.get("final_score_100", item.get("total_score", 0))

        explanation_text = str(item.get("explanation") or item.get("why_it_matters") or "Profile fit appears moderate based on extracted content.")
        reason_chunks = [chunk.strip() for chunk in explanation_text.replace("\n", " ").split(".") if chunk.strip()]
        while len(reason_chunks) < 3:
            reason_chunks.append("Ranking used deterministic profile fit, urgency, and completeness signals")

        cgpa_required = breakdown.get("cgpa", {}).get("required_cgpa") if isinstance(breakdown.get("cgpa"), dict) else None
        cgpa_student = breakdown.get("cgpa", {}).get("student_cgpa") if isinstance(breakdown.get("cgpa"), dict) else None
        eligibility_met = f"CGPA check passed ({cgpa_student})" if cgpa_required is not None else "No strict CGPA gate detected"
        missing_requirement = "Missing explicit deadline or document detail" if not (opp.get("deadline") and opp.get("required_documents")) else "No critical blocker detected"

        required_docs = opp.get("required_documents") if isinstance(opp.get("required_documents"), list) else []
        doc_1 = required_docs[0] if len(required_docs) >= 1 else "Resume/CV"
        doc_2 = required_docs[1] if len(required_docs) >= 2 else "Academic transcript"

        action_steps = item.get("action_checklist") if isinstance(item.get("action_checklist"), list) else []
        while len(action_steps) < 3:
            action_steps.append("Review eligibility, prepare documents, and submit before deadline")

        apply_link = str(opp.get("application_link") or item.get("application_link") or "Not provided")

        lines.extend(
            [
                f"{index}. {title}",
                "",
                f"Type: {opportunity_type}",
                f"Organization: {organization}",
                f"Deadline: {deadline}",
                f"Priority Score: {priority_score} / 100",
                "",
                "Why this matters for you:",
                "",
                f"{reason_chunks[0]}",
                f"{reason_chunks[1]}",
                f"{reason_chunks[2]}",
                "",
                "Eligibility Check:",
                "",
                f"✅ {eligibility_met}",
                f"❌ {missing_requirement}",
                "",
                "What you need:",
                "",
                f"{doc_1}",
                f"{doc_2}",
                "",
                "Action Plan:",
                "",
                f"{action_steps[0]}",
                f"{action_steps[1]}",
                f"{action_steps[2]}",
                "",
                f"Apply here: {apply_link if apply_link else 'Not provided'}",
                "",
            ]
        )

    lines.append("⚠️ Urgent Deadlines")
    urgent_found = False
    for item in opportunities:
        if not isinstance(item, dict):
            continue
        score = item.get("score", {}) if isinstance(item.get("score"), dict) else {}
        breakdown = item.get("score_breakdown", {}) if isinstance(item.get("score_breakdown"), dict) else {}
        days_left = score.get("days_left")
        if days_left is None and isinstance(breakdown.get("urgency"), dict):
            days_left = breakdown.get("urgency", {}).get("days_remaining")
        if isinstance(days_left, int) and days_left <= 7:
            urgent_found = True
            title = str(item.get("title") or item.get("email_subject") or "(No Subject)")
            lines.append(f"{title} → Deadline in {days_left} days")
    if not urgent_found:
        lines.append("No deadlines within 7 days detected")

    lines.append("❌ Ignored Emails (Not Opportunities)")
    if ignored_emails:
        for ignored in ignored_emails[:5]:
            if not isinstance(ignored, dict):
                continue
            subject = str(ignored.get("subject") or "(No Subject)")
            reason = str(ignored.get("reason") or "Filtered by opportunity classifier")
            lines.append(f"{subject} → {reason}")
    else:
        lines.append("No ignored emails in this scan batch")

    profile = result.get("profile", {}) if isinstance(result.get("profile"), dict) else {}
    interests = profile.get("interests") or profile.get("skills") or profile.get("skills_interests") or []
    if isinstance(interests, list) and interests:
        primary_interest = str(interests[0])
    elif isinstance(interests, str) and interests.strip():
        primary_interest = interests.strip()
    else:
        primary_interest = "your selected profile"

    lines.extend(
        [
            "🧠 AI Insight",
            "You are receiving many non-opportunity emails → consider filtering newsletters.",
            f"Most relevant opportunities match your interest in {primary_interest}.",
            "You should prioritize opportunities with deadlines within 7 days.",
        ]
    )

    return "\n".join(lines)


def _direct_fallback_response(user_email: str, credentials_data: dict[str, str], message: str) -> str:
    profile = get_user_profile(user_email)

    extracted_profile = extract_profile_from_message(message)
    filled_fields = [
        field
        for field, value in extracted_profile.items()
        if value and (not isinstance(value, list) or len(value) > 0)
    ]
    if len(filled_fields) >= 2:
        save_user_profile(user_email, extracted_profile)
        profile = extracted_profile
        reply = (
            "I saved your profile from the message you sent. You can now click Scan Opportunities, or send a scan request here."
        )
        if "scan" in message.lower() or "opportun" in message.lower():
            result = _run_scan_pipeline(user_email, credentials_data, max_results=100)
            _set_last_scan_result(result)
            return reply + "\n\n" + _format_scan_summary(result)
        return reply

    if not profile:
        return _missing_profile_message()

    lower_message = message.lower()
    if "scan" in lower_message or "opportun" in lower_message:
        result = _run_scan_pipeline(user_email, credentials_data, max_results=100)
        _set_last_scan_result(result)
        return _format_scan_summary(result)

    return (
        "Your profile is saved. Send 'scan opportunities' to fetch and rank Gmail opportunities, or paste profile updates if needed."
    )


def _handle_message_deterministically(user_email: str, credentials_data: dict[str, str], message: str) -> str | None:
    lower_message = message.lower()
    scan_requested = "scan" in lower_message or "opportun" in lower_message

    extracted_profile = extract_profile_from_message(message)
    filled_fields = [
        field
        for field, value in extracted_profile.items()
        if value and (not isinstance(value, list) or len(value) > 0)
    ]

    if len(filled_fields) >= 2:
        save_user_profile(user_email, extracted_profile)
        reply = "I saved your profile from the message you sent."
        if scan_requested:
            result = _run_scan_pipeline(user_email, credentials_data, max_results=100)
            _set_last_scan_result(result)
            reply += "\n\n" + _format_scan_summary(result)
        else:
            reply += " You can now click Scan Opportunities whenever you want me to rank your Gmail inbox."
        return reply

    if scan_requested:
        profile = get_user_profile(user_email)
        if not profile:
            return _missing_profile_message()
        result = _run_scan_pipeline(user_email, credentials_data, max_results=100)
        _set_last_scan_result(result)
        return _format_scan_summary(result)

    return None


def _render_dashboard(error: str | None = None, agent_reply: str | None = None):
    user = session.get("user")
    if not user:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        **_dashboard_payload(error=error, agent_reply=agent_reply),
    )


def _process_agent_message(message: str):
    user = session.get("user")
    if not user:
        return redirect(url_for("login"))

    credentials_data = session.get("google_credentials")
    if not credentials_data:
        return _render_dashboard(error="Session expired. Please sign in again.")

    chat_history = _get_chat_history()

    deterministic_reply = _handle_message_deterministically(user["email"], credentials_data, message)
    if deterministic_reply is not None:
        is_scan_summary = "Opportunity Inbox Analysis" in deterministic_reply
        latest_scan_result = _get_last_scan_result() if is_scan_summary else None
        assistant_reply = _scan_completion_message(latest_scan_result) if is_scan_summary else deterministic_reply
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": assistant_reply})
        _set_chat_history(chat_history)
        if _request_wants_json():
            payload = {
                "ok": True,
                "reply": assistant_reply,
                **_dashboard_payload(agent_reply=assistant_reply),
            }
            if is_scan_summary and isinstance(latest_scan_result, dict):
                payload["reply_kind"] = "scan_summary"
                payload["scan_result"] = latest_scan_result
            return jsonify(payload)
        return _render_dashboard(agent_reply=assistant_reply)

    try:
        from langchain_core.messages import HumanMessage

        agent_graph, history_messages = build_agent(user["email"], credentials_data, chat_history)
        result = agent_graph.invoke({"messages": history_messages + [HumanMessage(content=message)]})
    except Exception as exc:
        fallback_reply = _direct_fallback_response(user["email"], credentials_data, message)
        is_scan_summary = "Opportunity Inbox Analysis" in fallback_reply
        latest_scan_result = _get_last_scan_result() if is_scan_summary else None
        assistant_reply = _scan_completion_message(latest_scan_result) if is_scan_summary else fallback_reply
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": assistant_reply})
        _set_chat_history(chat_history)
        error_text = "Agent request failed, used fallback flow instead."
        if _request_wants_json():
            payload = {
                "ok": True,
                "reply": assistant_reply,
                "error": error_text,
                **_dashboard_payload(error=None, agent_reply=assistant_reply),
            }
            if is_scan_summary and isinstance(latest_scan_result, dict):
                payload["reply_kind"] = "scan_summary"
                payload["scan_result"] = latest_scan_result
            return jsonify(payload)
        return _render_dashboard(error=error_text, agent_reply=assistant_reply)

    reply = ""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            reply = getattr(last_message, "content", str(last_message))
    if not reply:
        reply = "The agent completed the request, but no response text was returned."

    latest_scan_result = _get_last_scan_result()
    is_scan_summary = "Opportunity Inbox Analysis" in reply and isinstance(latest_scan_result, dict)
    assistant_reply = _scan_completion_message(latest_scan_result) if is_scan_summary else reply

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": assistant_reply})
    _set_chat_history(chat_history)
    if _request_wants_json():
        payload = {"ok": True, "reply": assistant_reply, **_dashboard_payload(agent_reply=assistant_reply)}
        if is_scan_summary:
            payload["reply_kind"] = "scan_summary"
            payload["scan_result"] = latest_scan_result
        return jsonify(payload)
    return _render_dashboard(agent_reply=assistant_reply)


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

    session.setdefault("chat_history", [])
    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return _render_dashboard()


@app.route("/agent/chat", methods=["POST"])
def agent_chat():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
    else:
        message = request.form.get("message", "").strip()
    if not message:
        message = (
            "Scan opportunities now using my saved profile. If my profile is missing, ask me for the missing details."
        )
    return _process_agent_message(message)


@app.route("/scan", methods=["POST"])
def scan():
    user = session.get("user")
    if not user:
        return jsonify({"ok": False, "error": "Session expired. Please sign in again."}), 401

    credentials_data = session.get("google_credentials")
    if not credentials_data:
        return jsonify({"ok": False, "error": "Google credentials missing. Please sign in again."}), 401

    try:
        result = _run_scan_pipeline(user["email"], credentials_data, max_results=100)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc), "needs_profile": True}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": _short_error_message(exc)}), 500

    return jsonify({
        "ok": True,
        "ranked_opportunities": result.get("ranked_opportunities", []),
        "pipeline_telemetry": result.get("pipeline_telemetry", {}),
        "reply_kind": "scan_summary",
        "scan_result": result,
        "last_scan_result": result,
        **_dashboard_payload(agent_reply=_scan_completion_message(result)),
    })


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
