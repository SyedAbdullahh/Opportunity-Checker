import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, List

import httpx
from googleapiclient.discovery import build
from openai import OpenAI

from db import save_user_profile
from scoring import calculate_match_score

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]

BIG_ORGS = {
    "google",
    "microsoft",
    "meta",
    "amazon",
    "ibm",
    "unesco",
    "unicef",
    "world bank",
    "mit",
    "stanford",
}


def _clean_value(value: str) -> str:
    cleaned = value.strip().strip(". ,;:-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _append_unique(items: list[str], value: str) -> None:
    normalized = _clean_value(value)
    if not normalized:
        return
    lowered = normalized.lower()
    if lowered not in {item.lower() for item in items}:
        items.append(normalized)


def _to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]
    return []


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _openai_client() -> OpenAI:
    disable_ssl_verify = os.getenv("OPENAI_DISABLE_SSL_VERIFY", "true").lower() in {"1", "true", "yes", "on"}
    timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
    http_client = httpx.Client(verify=not disable_ssl_verify, timeout=timeout)
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=http_client)


def _llm_json(system_prompt: str, user_prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        client = _openai_client()
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = _safe_json(content)
        return parsed if parsed else fallback
    except Exception:
        return fallback


def _extract_header(headers: list[dict[str, Any]], key: str) -> str:
    key_lower = key.lower()
    for header in headers:
        if header.get("name", "").lower() == key_lower:
            return header.get("value", "")
    return ""


def _decode_base64(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _decode_body(payload: dict[str, Any]) -> str:
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


def _parse_internal_date(internal_date_ms: str) -> str:
    try:
        dt = datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)
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


def _parse_deadline_date(deadline: str | None) -> datetime | None:
    if not deadline:
        return None
    deadline = deadline.strip()
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(deadline, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _build_gmail_service(credentials_data: dict[str, Any]):
    from google.oauth2.credentials import Credentials

    credentials = Credentials(
        token=credentials_data.get("token"),
        refresh_token=credentials_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        scopes=SCOPES,
    )
    return build("gmail", "v1", credentials=credentials)


def _normalize_profile(profile_data: dict[str, Any] | None) -> dict[str, Any]:
    profile_data = profile_data or {}
    degree = str(profile_data.get("degree") or profile_data.get("degree_program") or "").strip()
    semester_raw = str(profile_data.get("semester") or "").strip()
    semester_match = re.search(r"\d+", semester_raw)
    semester = int(semester_match.group(0)) if semester_match else 0

    cgpa_raw = str(profile_data.get("cgpa") or "").strip()
    try:
        cgpa = float(cgpa_raw)
    except Exception:
        cgpa = 0.0

    skills = _to_list(profile_data.get("skills") or profile_data.get("skills_interests"))
    interests = _to_list(profile_data.get("interests"))
    preferred_types = _to_list(profile_data.get("preferred_types") or profile_data.get("preferred_opportunity_types"))

    financial_need_value = profile_data.get("financial_need")
    financial_need = False
    if isinstance(financial_need_value, bool):
        financial_need = financial_need_value
    elif isinstance(financial_need_value, str):
        financial_need = financial_need_value.lower() in {"true", "yes", "required", "high", "medium"}

    location_preference = str(profile_data.get("location_preference") or "").strip().lower()

    return {
        "degree": degree,
        "semester": semester,
        "cgpa": cgpa,
        "skills": skills,
        "interests": interests,
        "preferred_types": [item.lower() for item in preferred_types],
        "financial_need": financial_need,
        "location_preference": location_preference,
        "past_experience": str(profile_data.get("past_experience") or "").strip(),
    }


def classify_opportunity_from_title(title: str) -> dict[str, Any]:
    title_lower = title.lower()
    fallback_is_opportunity = any(
        token in title_lower
        for token in [
            "opportunity",
            "opportunities",
            "apply",
            "application",
            "scholarship",
            "internship",
            "competition",
            "fellowship",
            "exchange",
            "summer school",
            "grant",
            "admission",
        ]
    )
    fallback = {
        "is_opportunity": fallback_is_opportunity,
        "type": "other",
    }
    if "scholarship" in title_lower:
        fallback["type"] = "scholarship"
    elif "internship" in title_lower:
        fallback["type"] = "internship"
    elif "competition" in title_lower or "hackathon" in title_lower:
        fallback["type"] = "competition"
    elif "fellowship" in title_lower:
        fallback["type"] = "fellowship"
    elif "exchange" in title_lower or "summer school" in title_lower or "admission" in title_lower:
        fallback["type"] = "other"

    system_prompt = (
        "Classify whether an email title indicates a real student opportunity. "
        "Return strict JSON only with keys: is_opportunity (bool), "
        "type (scholarship|internship|competition|fellowship|other)."
    )
    user_prompt = f"Email title: {title}"
    result = _llm_json(system_prompt, user_prompt, fallback)

    llm_is_opportunity = bool(result.get("is_opportunity", fallback["is_opportunity"]))
    llm_type = str(result.get("type", fallback["type"])).lower()

    return {
        "is_opportunity": llm_is_opportunity,
        "type": llm_type,
    }


def _extract_deadline_heuristic(text: str) -> str:
    patterns = [
        r"deadline[:\s]+([A-Za-z0-9,\-/ ]{6,30})",
        r"apply by[:\s]+([A-Za-z0-9,\-/ ]{6,30})",
        r"last date[:\s]+([A-Za-z0-9,\-/ ]{6,30})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_value(match.group(1))
    return ""


def _extract_link_heuristic(text: str) -> str:
    match = re.search(r"https?://[^\s)]+", text)
    return match.group(0) if match else ""


def extract_structured_opportunity(title: str, body: str, opportunity_type: str) -> dict[str, Any]:
    fallback = {
        "title": title,
        "organization": "",
        "opportunity_type": opportunity_type or "other",
        "deadline": _extract_deadline_heuristic(body),
        "eligibility_criteria": [],
        "required_documents": [],
        "location": "",
        "benefits": [],
        "application_link": _extract_link_heuristic(body),
        "contact_info": "",
    }

    system_prompt = (
        "Extract structured opportunity info from an email. "
        "Return strict JSON only with keys: title, organization, opportunity_type, deadline, "
        "eligibility_criteria (array), required_documents (array), location, benefits (array), "
        "application_link, contact_info."
    )
    user_prompt = f"Email title:\n{title}\n\nEmail body:\n{body[:8000]}"
    parsed = _llm_json(system_prompt, user_prompt, fallback)

    return {
        "title": str(parsed.get("title") or fallback["title"]),
        "organization": str(parsed.get("organization") or ""),
        "opportunity_type": str(parsed.get("opportunity_type") or fallback["opportunity_type"]).lower(),
        "deadline": str(parsed.get("deadline") or fallback["deadline"]),
        "eligibility_criteria": _to_list(parsed.get("eligibility_criteria")),
        "required_documents": _to_list(parsed.get("required_documents")),
        "location": str(parsed.get("location") or ""),
        "benefits": _to_list(parsed.get("benefits")),
        "application_link": str(parsed.get("application_link") or fallback["application_link"]),
        "contact_info": str(parsed.get("contact_info") or ""),
    }


def _profile_fit_score(opportunity: dict[str, Any], profile: dict[str, Any], raw_text: str) -> float:
    fit = 0.0

    degree_match = 0.0
    if profile["degree"]:
        if profile["degree"].lower() in raw_text.lower() or any(profile["degree"].lower() in item.lower() for item in opportunity.get("eligibility_criteria", [])):
            degree_match = 0.3
    fit += degree_match

    skills_match = 0.0
    if profile["skills"]:
        overlap = [skill for skill in profile["skills"] if skill.lower() in raw_text.lower()]
        if overlap:
            ratio = min(1.0, len(overlap) / max(1, len(profile["skills"])))
            skills_match = 0.4 * ratio
    fit += skills_match

    eligibility_match = 0.0
    if opportunity.get("eligibility_criteria"):
        criteria_text = " ".join(opportunity.get("eligibility_criteria", [])).lower()
        if profile["semester"]:
            sem_match = re.search(r"(\d+)(?:st|nd|rd|th)?\s*(?:year|semester)\+?", criteria_text)
            if sem_match:
                required = int(sem_match.group(1))
                if profile["semester"] >= required:
                    eligibility_match = max(eligibility_match, 0.15)
        if profile["cgpa"]:
            cgpa_match = re.search(r"cgpa\s*(?:>=|>|at least|minimum)?\s*(\d(?:\.\d+)?)", criteria_text)
            if cgpa_match:
                required_cgpa = float(cgpa_match.group(1))
                if profile["cgpa"] >= required_cgpa:
                    eligibility_match = max(eligibility_match, 0.15)
        if not eligibility_match:
            eligibility_match = 0.1
    fit += min(0.3, eligibility_match)

    return max(0.0, min(1.0, fit))


def _urgency_score(deadline_text: str) -> tuple[float, int | None]:
    deadline = _parse_deadline_date(deadline_text)
    if not deadline:
        return 0.2, None

    today = datetime.now(timezone.utc)
    days_left = (deadline.date() - today.date()).days

    if days_left <= 2:
        return 1.0, days_left
    if days_left <= 7:
        return 0.8, days_left
    if days_left <= 14:
        return 0.5, days_left
    return 0.2, days_left


def _opportunity_value_score(opportunity: dict[str, Any], profile: dict[str, Any], raw_text: str) -> float:
    value = 0.0
    full_text = raw_text.lower()

    if any(term in full_text for term in ["stipend", "salary", "funded", "paid", "grant amount", "allowance"]):
        value += 0.5

    org = opportunity.get("organization", "").lower()
    if any(name in org or name in full_text for name in BIG_ORGS):
        value += 0.3

    opp_type = opportunity.get("opportunity_type", "").lower()
    if opp_type in profile["preferred_types"]:
        value += 0.2
    elif any(interest.lower() in full_text for interest in profile["interests"]):
        value += 0.2

    return max(0.0, min(1.0, value))


def _completeness_score(opportunity: dict[str, Any]) -> float:
    score = 0.0
    if opportunity.get("deadline"):
        score += 0.3
    if opportunity.get("application_link"):
        score += 0.3
    if opportunity.get("required_documents"):
        score += 0.4
    return max(0.0, min(1.0, score))


def _score_opportunity(opportunity: dict[str, Any], profile: dict[str, Any], raw_text: str) -> dict[str, Any]:
    profile_fit = _profile_fit_score(opportunity, profile, raw_text)
    urgency, days_left = _urgency_score(opportunity.get("deadline", ""))
    value = _opportunity_value_score(opportunity, profile, raw_text)
    completeness = _completeness_score(opportunity)

    final_score = (
        0.35 * profile_fit
        + 0.30 * urgency
        + 0.20 * value
        + 0.15 * completeness
    )

    return {
        "final_score": round(final_score, 4),
        "final_score_100": round(final_score * 100, 1),
        "profile_fit": round(profile_fit, 4),
        "urgency": round(urgency, 4),
        "opportunity_value": round(value, 4),
        "completeness": round(completeness, 4),
        "days_left": days_left,
    }


def generate_action_checklist(opportunity: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    for doc in opportunity.get("required_documents", []):
        _append_unique(steps, f"Prepare {doc}")

    if opportunity.get("deadline"):
        _append_unique(steps, f"Apply before {opportunity['deadline']}")

    if opportunity.get("application_link"):
        _append_unique(steps, "Open and complete the application form")

    if opportunity.get("contact_info"):
        _append_unique(steps, f"Contact for clarification: {opportunity['contact_info']}")

    if not steps:
        steps = [
            "Review opportunity details carefully",
            "Prepare your resume and supporting documents",
            "Submit application before the deadline",
        ]
    return steps


def generate_ranking_explanation(opportunity: dict[str, Any], score: dict[str, Any], profile: dict[str, Any]) -> str:
    fallback = (
        f"Ranked high due to profile fit ({score['profile_fit']:.2f}), urgency ({score['urgency']:.2f}), "
        f"opportunity value ({score['opportunity_value']:.2f}), and completeness ({score['completeness']:.2f})."
    )

    system_prompt = (
        "Explain why this opportunity was ranked for this student. "
        "Keep it concise and practical. Mention fit reasons, urgency, benefits, and missing requirements."
    )
    user_prompt = json.dumps(
        {
            "student_profile": profile,
            "opportunity": opportunity,
            "score_components": score,
        },
        ensure_ascii=False,
    )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        client = _openai_client()
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        return text or fallback
    except Exception:
        return fallback


def scan_gmail_for_opportunities(
    user_email: str,
    credentials_data: dict[str, Any],
    profile_data: dict[str, Any] | None = None,
    max_results: int = 30,
) -> dict[str, Any]:
    gmail = _build_gmail_service(credentials_data)
    profile = _normalize_profile(profile_data)

    messages_result = gmail.users().messages().list(
        userId="me",
        q="newer_than:365d -in:spam -in:trash",
        maxResults=max_results,
    ).execute()

    messages = messages_result.get("messages", [])
    ranked: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []

    for index, message in enumerate(messages, start=1):
        detail = gmail.users().messages().get(
            userId="me",
            id=message["id"],
            format="full",
        ).execute()

        payload = detail.get("payload", {})
        headers = payload.get("headers", [])
        subject = _extract_header(headers, "Subject") or "(No Subject)"
        sender = _extract_header(headers, "From") or "(Unknown Sender)"
        date_header = _extract_header(headers, "Date")
        body_text = _decode_body(payload)
        snippet = detail.get("snippet", "")

        print(f"[SCAN] Email {index}/{len(messages)}: {subject}", flush=True)

        detection = classify_opportunity_from_title(subject)
        if not detection["is_opportunity"]:
            ignored.append(
                {
                    "subject": subject,
                    "from": sender,
                    "date": _parse_date_header(date_header) or _parse_internal_date(detail.get("internalDate", "")),
                    "reason": "LLM title classification = not an opportunity",
                }
            )
            continue

        raw_text = f"{subject}\n{snippet}\n{body_text}"
        structured = extract_structured_opportunity(subject, raw_text, detection["type"])
        score = _score_opportunity(structured, profile, raw_text)
        explanation = generate_ranking_explanation(structured, score, profile)
        checklist = generate_action_checklist(structured)

        ranked.append(
            {
                "rank": index,
                "email_subject": subject,
                "email_from": sender,
                "email_date": _parse_date_header(date_header) or _parse_internal_date(detail.get("internalDate", "")),
                "detection": detection,
                "opportunity": structured,
                "score": score,
                "explanation": explanation,
                "action_checklist": checklist,
                "evidence_snippet": snippet[:280],
            }
        )

    ranked.sort(key=lambda item: item["score"]["final_score"], reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx

    return {
        "user_email": user_email,
        "profile": profile,
        "scanned_count": len(messages),
        "opportunities_detected": len(ranked),
        "ignored_count": len(ignored),
        "ranked_opportunities": ranked,
        "ignored_emails": ignored,
    }


def extract_profile_from_message(profile_message: str) -> dict[str, Any]:
    text = profile_message.strip()
    lowered = text.lower()
    profile: dict[str, Any] = {
        "degree_program": "",
        "semester": "",
        "cgpa": "",
        "skills_interests": [],
        "preferred_opportunity_types": [],
        "financial_need": "",
        "location_preference": "",
        "past_experience": "",
    }

    # Parse explicit "Label: value" lines first (user-friendly form format).
    label_aliases = {
        "degree program": "degree_program",
        "degree_program": "degree_program",
        "degree": "degree_program",
        "current semester": "semester",
        "semester": "semester",
        "cgpa": "cgpa",
        "skills/interests": "skills_interests",
        "skills_interests": "skills_interests",
        "skills": "skills_interests",
        "interests": "skills_interests",
        "preferred opportunity types": "preferred_opportunity_types",
        "preferred_opportunity_types": "preferred_opportunity_types",
        "opportunity types": "preferred_opportunity_types",
        "financial need required (yes/no)": "financial_need",
        "financial need": "financial_need",
        "financial_need": "financial_need",
        "location preference": "location_preference",
        "location_preference": "location_preference",
        "location": "location_preference",
        "past experience": "past_experience",
        "past_experience": "past_experience",
        "experience": "past_experience",
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        key_part, value_part = line.split(":", 1)
        key = key_part.strip().lower()
        value = _clean_value(value_part)
        if not value:
            continue

        canonical = label_aliases.get(key)
        if not canonical:
            continue

        if canonical == "degree_program":
            profile["degree_program"] = value
        elif canonical == "semester":
            profile["semester"] = value
        elif canonical == "cgpa":
            cgpa_in_value = re.search(r"\b(\d(?:\.\d+)?)\b", value)
            if cgpa_in_value:
                profile["cgpa"] = cgpa_in_value.group(1)
        elif canonical == "skills_interests":
            for item in re.split(r",|/|\bor\b", value, flags=re.IGNORECASE):
                _append_unique(profile["skills_interests"], item)
        elif canonical == "preferred_opportunity_types":
            for item in re.split(r",|/|\bor\b", value, flags=re.IGNORECASE):
                normalized = _clean_value(item).lower()
                if normalized in {"scholarship", "internship", "fellowship", "competition", "admission", "grant"}:
                    _append_unique(profile["preferred_opportunity_types"], normalized)
        elif canonical == "financial_need":
            lower_value = value.lower()
            if lower_value in {"yes", "y", "true", "required", "need", "high", "medium"}:
                profile["financial_need"] = "required"
            elif lower_value in {"no", "n", "false", "not required", "not_required"}:
                profile["financial_need"] = "not_required"
        elif canonical == "location_preference":
            profile["location_preference"] = value
        elif canonical == "past_experience":
            profile["past_experience"] = value

    semester_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*semester\b", lowered, re.IGNORECASE)
    if semester_match and not profile["semester"]:
        profile["semester"] = f"{semester_match.group(1)} semester"

    cgpa_match = re.search(r"\b(\d(?:\.\d+)?)\s*cgpa\b", lowered, re.IGNORECASE)
    if cgpa_match and not profile["cgpa"]:
        profile["cgpa"] = cgpa_match.group(1)

    degree_patterns = [
        r"\b(bs\s*[a-z0-9&/-]{0,20})\b",
        r"\b(bsc\s*[a-z0-9&/-]{0,20})\b",
        r"\b(ms\s*[a-z0-9&/-]{0,20})\b",
        r"\b(msc\s*[a-z0-9&/-]{0,20})\b",
        r"\b(bachelor(?:'s)?\s+of\s+[a-z\s]{3,30})\b",
        r"\b(master(?:'s)?\s+of\s+[a-z\s]{3,30})\b",
    ]
    for pattern in degree_patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match and not profile["degree_program"]:
            profile["degree_program"] = _clean_value(match.group(1))
            break

    if not profile["degree_program"]:
        leading_clause = text.split(",")[0].strip()
        if any(token in lowered for token in ["semester", "cgpa", "student", "study", "se ", "cs ", "ai "]):
            profile["degree_program"] = _clean_value(leading_clause)

    if not profile["financial_need"] and any(term in lowered for term in ["financial need", "financial support", "need required", "need yes"]):
        profile["financial_need"] = "required"

    location_patterns = [
        r"\b([a-z][a-z\s.-]{2,30})\s+is\s+preferred\b",
        r"location(?:\s*[:=]\s*|\s+prefer(?:red)?:?\s*)([a-z][a-z\s.-]{2,30})",
    ]
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and not profile["location_preference"]:
            profile["location_preference"] = _clean_value(match.group(1))
            break

    experience_match = re.search(r"experience(?:\s*[:=]\s*|\s+)([^,;\n]{3,80})", text, re.IGNORECASE)
    if experience_match and not profile["past_experience"]:
        profile["past_experience"] = _clean_value(experience_match.group(1))

    for keyword in ["ai", "ml", "machine learning", "data science", "web", "cybersecurity", "research", "software"]:
        if keyword in lowered:
            _append_unique(profile["skills_interests"], keyword.upper() if keyword in {"ai", "ml"} else keyword.title())

    for option in ["scholarship", "internship", "fellowship", "competition", "admission", "grant"]:
        if option in lowered:
            _append_unique(profile["preferred_opportunity_types"], option)

    if not profile["skills_interests"]:
        skill_match = re.search(r"(?:skills?|interests?)\s*[:=]\s*([^,;\n]{3,80})", text, re.IGNORECASE)
        if skill_match:
            for item in re.split(r"[,/]", skill_match.group(1)):
                _append_unique(profile["skills_interests"], item)

    return profile


def store_profile_from_message(user_email: str, profile_message: str) -> dict[str, Any]:
    profile = extract_profile_from_message(profile_message)
    save_user_profile(user_email, profile)
    return profile


def _extract_required_cgpa(text: str) -> float | None:
    match = re.search(r"(?:required|min(?:imum)?|at least)?\s*cgpa\s*(?:>=|>|:|is)?\s*(\d(?:\.\d+)?)", text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _extract_required_skills(text: str) -> list[str]:
    known_skills = [
        "python",
        "java",
        "javascript",
        "typescript",
        "react",
        "node",
        "sql",
        "machine learning",
        "ai",
        "data science",
        "cybersecurity",
        "cloud",
    ]
    lower_text = text.lower()
    skills: list[str] = []
    for skill in known_skills:
        if skill in lower_text:
            _append_unique(skills, skill)
    return skills


def extract_opportunity_from_email(email_text: str) -> dict[str, Any]:
    """
    Extract opportunity attributes from a raw email string.
    Returns a flat dict that includes is_opportunity for spam filtering.
    """
    text = (email_text or "").strip()
    title = text.splitlines()[0][:180] if text else "(No Subject)"

    keyword_hit = any(
        token in text.lower()
        for token in [
            "opportunity",
            "scholarship",
            "internship",
            "fellowship",
            "competition",
            "grant",
            "apply",
        ]
    )
    fallback = {
        "is_opportunity": keyword_hit,
        "title": title,
        "organization": "",
        "opportunity_type": "other",
        "deadline": "",
        "required_cgpa": _extract_required_cgpa(text),
        "required_skills": _extract_required_skills(text),
        "eligibility_criteria": [],
        "required_documents": [],
        "location": "",
        "benefits": [],
        "application_link": _extract_link_heuristic(text),
        "contact_info": "",
    }

    system_prompt = (
        "Extract opportunity details from the email text and classify if this is a real student opportunity. "
        "Return strict JSON only with keys: is_opportunity (bool), title, organization, opportunity_type, deadline, "
        "required_cgpa (number or null), required_skills (array), eligibility_criteria (array), required_documents (array), "
        "location, benefits (array), application_link, contact_info. "
        "Set is_opportunity=false for announcements/spam/non-opportunity messages."
    )
    parsed = _llm_json(system_prompt, text[:12000], fallback)

    required_cgpa = _to_float(parsed.get("required_cgpa"))
    if required_cgpa is None:
        required_cgpa = fallback["required_cgpa"]

    return {
        "is_opportunity": bool(parsed.get("is_opportunity", fallback["is_opportunity"])),
        "title": str(parsed.get("title") or fallback["title"]),
        "organization": str(parsed.get("organization") or fallback["organization"]),
        "opportunity_type": str(parsed.get("opportunity_type") or fallback["opportunity_type"]).lower(),
        "deadline": str(parsed.get("deadline") or fallback["deadline"]),
        "required_cgpa": required_cgpa,
        "required_skills": _to_list(parsed.get("required_skills") or fallback["required_skills"]),
        "eligibility_criteria": _to_list(parsed.get("eligibility_criteria") or fallback["eligibility_criteria"]),
        "required_documents": _to_list(parsed.get("required_documents") or fallback["required_documents"]),
        "location": str(parsed.get("location") or fallback["location"]),
        "benefits": _to_list(parsed.get("benefits") or fallback["benefits"]),
        "application_link": str(parsed.get("application_link") or fallback["application_link"]),
        "contact_info": str(parsed.get("contact_info") or fallback["contact_info"]),
    }


def _build_top3_explainer(opportunity: dict[str, Any], profile_data: Any, score_breakdown: dict[str, Any]) -> dict[str, Any]:
    fallback = {
        "why_it_matters": "This opportunity matches parts of your profile and score components. Review the details and apply if eligible.",
        "action_checklist": [
            "Review eligibility and required skills",
            "Prepare your documents",
            "Submit the application before deadline",
        ],
    }

    model = "gpt-4o-mini"
    system_prompt = (
        "You are an assistant that explains why an opportunity matters to a student. "
        "Return strict JSON only with keys: why_it_matters (string, exactly 2 sentences), "
        "action_checklist (array of exactly 3 short strings)."
    )
    user_prompt = json.dumps(
        {
            "opportunity": opportunity,
            "student_profile": profile_data,
            "score_breakdown": score_breakdown,
        },
        ensure_ascii=False,
    )

    try:
        client = _openai_client()
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = _safe_json(response.choices[0].message.content or "{}")
        why_text = str(parsed.get("why_it_matters") or fallback["why_it_matters"]).strip()
        checklist = _to_list(parsed.get("action_checklist"))
        if len(checklist) < 3:
            checklist = fallback["action_checklist"]
        return {
            "why_it_matters": why_text,
            "action_checklist": checklist[:3],
        }
    except Exception as exc:
        print(f"[PIPELINE] Explainer fallback used due to error: {exc}", flush=True)
        return fallback


def process_and_rank_inbox(
    raw_emails: List[str],
    profile_data: Any,
    telemetry: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Pipeline: extract -> spam filter -> score -> rank -> explain top 3.
    """
    ranked_opportunities: list[dict[str, Any]] = []
    total_emails = len(raw_emails or [])
    telemetry_data = telemetry if telemetry is not None else {}
    telemetry_data.setdefault("emails_fetched", total_emails)
    telemetry_data.setdefault("fetch_time", 0.0)
    telemetry_data.setdefault("extraction_time", 0.0)
    telemetry_data.setdefault("scoring_time", 0.0)
    telemetry_data.setdefault("explainer_time", 0.0)
    telemetry_data.setdefault("opportunity_candidates", 0)
    telemetry_data.setdefault("non_opportunity_skipped", 0)
    telemetry_data.setdefault("ranked_opportunities_count", 0)

    print(f"[PIPELINE] Starting inbox processing for {total_emails} emails", flush=True)

    for index, email_text in enumerate(raw_emails or [], start=1):
        print(f"[PIPELINE] [{index}/{total_emails}] Extracting opportunity data", flush=True)
        try:
            extraction_started = time.time()
            extracted = extract_opportunity_from_email(email_text)
            telemetry_data["extraction_time"] = round(telemetry_data.get("extraction_time", 0.0) + (time.time() - extraction_started), 3)
        except Exception as exc:
            print(f"[PIPELINE] [{index}/{total_emails}] Extraction failed: {exc}", flush=True)
            continue

        if not extracted.get("is_opportunity", False):
            print(f"[PIPELINE] [{index}/{total_emails}] Skipped (not an opportunity)", flush=True)
            telemetry_data["non_opportunity_skipped"] = int(telemetry_data.get("non_opportunity_skipped", 0)) + 1
            continue

        telemetry_data["opportunity_candidates"] = int(telemetry_data.get("opportunity_candidates", 0)) + 1

        print(f"[PIPELINE] [{index}/{total_emails}] Scoring opportunity", flush=True)
        scoring_started = time.time()
        score_result = calculate_match_score(profile_data, extracted)
        telemetry_data["scoring_time"] = round(telemetry_data.get("scoring_time", 0.0) + (time.time() - scoring_started), 3)
        combined = {
            **extracted,
            "total_score": int(score_result.get("total_score", 0)),
            "is_eligible": bool(score_result.get("is_eligible", True)),
            "score_breakdown": score_result.get("breakdown", {}),
        }
        ranked_opportunities.append(combined)
        telemetry_data["ranked_opportunities_count"] = int(telemetry_data.get("ranked_opportunities_count", 0)) + 1
        print(
            f"[PIPELINE] [{index}/{total_emails}] Added: {combined.get('title', '(No Title)')} | score={combined['total_score']}",
            flush=True,
        )

    ranked_opportunities.sort(key=lambda item: int(item.get("total_score", 0)), reverse=True)
    print(f"[PIPELINE] Ranking completed. Total ranked opportunities: {len(ranked_opportunities)}", flush=True)

    for top_index, item in enumerate(ranked_opportunities[:3], start=1):
        print(f"[PIPELINE] Generating explainer for top {top_index}: {item.get('title', '(No Title)')}", flush=True)
        explainer_started = time.time()
        explainer = _build_top3_explainer(item, profile_data, item.get("score_breakdown", {}))
        telemetry_data["explainer_time"] = round(telemetry_data.get("explainer_time", 0.0) + (time.time() - explainer_started), 3)
        item["why_it_matters"] = explainer.get("why_it_matters", "")
        item["action_checklist"] = explainer.get("action_checklist", [])

    print("[PIPELINE] Process and rank pipeline finished", flush=True)
    return ranked_opportunities, telemetry_data
