from __future__ import annotations

from datetime import date, datetime
from typing import Any


def _get_value(source: Any, field: str, default: Any = None) -> Any:
    """Safely read a field from either a dict-like object or an attribute-based object."""
    if source is None:
        return default

    if isinstance(source, dict):
        return source.get(field, default)

    return getattr(source, field, default)


def _to_float(value: Any) -> float | None:
    """Convert incoming value to float, returning None for invalid/empty values."""
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


def _to_lower_set(values: Any) -> set[str]:
    """Normalize a list-like value into a lowercase string set."""
    if values is None:
        return set()

    if isinstance(values, str):
        values_iterable = [values]
    elif isinstance(values, (list, tuple, set)):
        values_iterable = values
    else:
        return set()

    normalized: set[str] = set()
    for item in values_iterable:
        if item is None:
            continue
        text = str(item).strip().lower()
        if text:
            normalized.add(text)
    return normalized


def _has_any_token(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def _urgency_points(deadline_value: Any) -> tuple[int, int | None]:
    """
    Return urgency points and days remaining.

    Rules:
    - <= 7 days: +20
    - <= 14 days: +10
    - invalid/empty/past deadline: +0
    """
    if deadline_value is None:
        return 0, None

    deadline_text = str(deadline_value).strip()
    if not deadline_text:
        return 0, None

    try:
        deadline_date = datetime.strptime(deadline_text, "%Y-%m-%d").date()
    except ValueError:
        return 0, None

    days_remaining = (deadline_date - date.today()).days
    if days_remaining < 0:
        return 0, days_remaining
    if days_remaining <= 7:
        return 20, days_remaining
    if days_remaining <= 14:
        return 10, days_remaining
    return 0, days_remaining


def calculate_match_score(profile: Any, opportunity: Any) -> dict[str, Any]:
    """
    Deterministically score how relevant an opportunity is for a student profile.

    Returns:
    {
      "total_score": int,
      "is_eligible": bool,
      "breakdown": dict
    }
    """
    total_score = 0

    breakdown: dict[str, Any] = {
        "cgpa": {"points": 0, "required_cgpa": None, "student_cgpa": None, "eligible": True},
        "skills_match": {"points": 0, "matched_skills": []},
        "type_match": {"points": 0, "matched": False},
        "location_match": {"points": 0, "matched": False},
        "financial_need_match": {"points": 0, "matched": False},
        "completeness": {"points": 0, "has_apply_path": False, "has_required_documents": False},
        "urgency": {"points": 0, "deadline": None, "days_remaining": None},
    }

    student_cgpa = _to_float(_get_value(profile, "cgpa"))
    required_cgpa = _to_float(_get_value(opportunity, "required_cgpa"))

    breakdown["cgpa"]["required_cgpa"] = required_cgpa
    breakdown["cgpa"]["student_cgpa"] = student_cgpa

    # CGPA hard eligibility gate.
    if required_cgpa is not None and student_cgpa is not None and student_cgpa < required_cgpa:
        breakdown["cgpa"]["eligible"] = False
        return {
            "total_score": 0,
            "is_eligible": False,
            "breakdown": breakdown,
        }

    # If there is no disqualifying CGPA condition, profile remains eligible and gets +15.
    total_score += 15
    breakdown["cgpa"]["points"] = 15

    student_skills = _to_lower_set(_get_value(profile, "skills"))
    required_skills = _to_lower_set(_get_value(opportunity, "required_skills"))

    matched_skills = sorted(student_skills.intersection(required_skills))
    skills_points = 10 * len(matched_skills)
    total_score += skills_points
    breakdown["skills_match"]["points"] = skills_points
    breakdown["skills_match"]["matched_skills"] = matched_skills

    opportunity_type_raw = _get_value(opportunity, "opportunity_type", "")
    opportunity_type = str(opportunity_type_raw).strip().lower() if opportunity_type_raw is not None else ""
    preferred_types = _to_lower_set(_get_value(profile, "preferred_types"))

    if opportunity_type and opportunity_type in preferred_types:
        total_score += 20
        breakdown["type_match"]["points"] = 20
        breakdown["type_match"]["matched"] = True

    # Location preference match (+10)
    profile_location_raw = _get_value(profile, "location_preference", "")
    opportunity_location_raw = _get_value(opportunity, "location", "")
    profile_location = str(profile_location_raw).strip().lower() if profile_location_raw is not None else ""
    opportunity_location = str(opportunity_location_raw).strip().lower() if opportunity_location_raw is not None else ""

    if profile_location and opportunity_location and (
        profile_location in opportunity_location or opportunity_location in profile_location
    ):
        total_score += 10
        breakdown["location_match"]["points"] = 10
        breakdown["location_match"]["matched"] = True

    # Financial need / stipend match (+10)
    profile_financial_need_raw = _get_value(profile, "financial_need", False)
    if isinstance(profile_financial_need_raw, bool):
        profile_financial_need = profile_financial_need_raw
    else:
        profile_financial_need = str(profile_financial_need_raw).strip().lower() in {
            "true", "yes", "required", "need", "high", "medium"
        }

    benefits = _get_value(opportunity, "benefits", [])
    benefit_text = " ".join(str(x) for x in benefits) if isinstance(benefits, (list, tuple, set)) else str(benefits or "")
    stipend_source_text = " ".join(
        [
            str(_get_value(opportunity, "title", "") or ""),
            str(_get_value(opportunity, "description", "") or ""),
            str(_get_value(opportunity, "eligibility_criteria", "") or ""),
            benefit_text,
        ]
    )

    stipend_tokens = ["stipend", "funded", "financial aid", "allowance", "tuition waiver", "paid", "grant"]

    if profile_financial_need and _has_any_token(stipend_source_text, stipend_tokens):
        total_score += 10
        breakdown["financial_need_match"]["points"] = 10
        breakdown["financial_need_match"]["matched"] = True

    # Completeness match (+10)
    application_link = str(_get_value(opportunity, "application_link", "") or "").strip()
    contact_info = str(_get_value(opportunity, "contact_info", "") or "").strip()
    required_documents = _get_value(opportunity, "required_documents", [])

    has_apply_path = bool(application_link or contact_info)
    has_required_documents = bool(required_documents) if isinstance(required_documents, (list, tuple, set)) else bool(str(required_documents).strip())

    breakdown["completeness"]["has_apply_path"] = has_apply_path
    breakdown["completeness"]["has_required_documents"] = has_required_documents
    if has_apply_path and has_required_documents:
        total_score += 10
        breakdown["completeness"]["points"] = 10

    deadline_value = _get_value(opportunity, "deadline")
    urgency_points, days_remaining = _urgency_points(deadline_value)
    total_score += urgency_points
    breakdown["urgency"]["points"] = urgency_points
    breakdown["urgency"]["deadline"] = None if deadline_value is None else str(deadline_value)
    breakdown["urgency"]["days_remaining"] = days_remaining

    return {
        "total_score": int(total_score),
        "is_eligible": True,
        "breakdown": breakdown,
    }
