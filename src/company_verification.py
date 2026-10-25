"""
==================================================================
Day 6: Company Verification and Fraud-Risk Analysis (Supplementary)
Project: AI-Based Fake Job Detection and Verification System
==================================================================

WHAT THIS MODULE DOES:
Takes three pieces of information a job seeker might type in
(company name, company website, application email) and performs
simple, fully explainable STRUCTURAL checks on them — never a
real-world identity check. It produces a 0-100 "verification risk
score", a Low/Medium/High level, and a plain-English list of the
signals that contributed to that score.

WHAT THIS MODULE DELIBERATELY DOES NOT DO:
    - It does NOT fetch, visit, or scrape the company website.
    - It does NOT call any external API or company registry.
    - It does NOT touch a database.
    - It does NOT use the trained fraud-detection model, TF-IDF, or
      any part of src/prediction.py — this is a completely separate,
      independent set of checks, meant to be shown ALONGSIDE the ML
      result, not merged into it.
    - It does NOT claim a company is "definitely genuine" or
      "definitely fraudulent". Every signal here is a WEAK, WARNING-
      LEVEL indicator, not proof.

WHY THIS IS A SEPARATE MODULE (NOT ADDED TO prediction.py):
prediction.py already has one clear job: take a job posting and run
it through the trained ML pipeline. Company verification is a
different KIND of check — rule-based, not model-based — that
happens to use different input fields (company_name,
company_website, application_email vs. the job posting text/
metadata). Keeping them separate makes each module easier to explain
on its own in a viva, and means neither file needs to change to
support the other.
==================================================================
"""

import re
from urllib.parse import urlparse

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# A short, well-known list of free/public email providers. This is
# intentionally simple and easy to extend — it is NOT an exhaustive
# real-world list, just enough to demonstrate the concept clearly.
FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "zoho.com", "gmx.com",
    "yandex.com", "live.com", "msn.com", "rediffmail.com",
}

# A very simple, widely-used email structure check: something@something.tld
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

# Risk points contributed by each signal (see explanation given to the
# user alongside this code for the reasoning behind each weight).
POINTS_NAME_MISSING = 25
POINTS_WEBSITE_MISSING = 20
POINTS_WEBSITE_INVALID = 15
POINTS_EMAIL_MISSING = 20
POINTS_EMAIL_INVALID = 15
POINTS_FREE_EMAIL = 15
POINTS_DOMAIN_MISMATCH = 20
POINTS_DOMAIN_MATCH_BONUS = -10  # reduces risk; a reassuring signal

VERIFICATION_LEVEL_THRESHOLDS = {"Low Risk": (0, 29), "Medium Risk": (30, 59), "High Risk": (60, 100)}

DISCLAIMER = (
    "This verification is based only on the company name, website, and email "
    "address supplied by the user. It performs simple structural checks and "
    "does NOT confirm a company's real-world identity, legitimacy, or "
    "registration status. No website was visited and no external service was "
    "contacted. A missing field, a free email provider, or a domain mismatch "
    "is a WARNING SIGNAL only — none of these, alone or combined, prove that "
    "a job posting is fraudulent."
)


# ==================================================================
# STEP 1 — SIMPLE PRESENCE CHECKS
# ==================================================================
def is_provided(value) -> bool:
    """
    WHAT: Checks whether a value counts as "provided" — not None,
          not missing, and not just whitespace after stripping.
    WHY:  A single, consistent definition of "missing" avoids
          subtle bugs where one part of the module treats "" as
          provided and another doesn't.
    """
    return value is not None and str(value).strip() != ""


# ==================================================================
# STEP 2 — WEBSITE URL STRUCTURE CHECK
# ==================================================================
def validate_website_structure(website: str):
    """
    WHAT: Checks whether the given website string LOOKS like a valid
          URL/domain — it does NOT check whether the site actually
          exists or is reachable (no network request is made).
    HOW: Accepts input with or without a scheme (e.g. both
          "example.com" and "https://example.com" are accepted). A
          missing scheme is assumed to be "http://" purely so
          Python's urlparse can extract the domain part; this does
          NOT mean we ever connect to the site.
    RETURNS: (is_valid: bool, domain: str or None)
    """
    if not is_provided(website):
        return False, None

    candidate = website.strip()
    if "://" not in candidate:
        candidate = "http://" + candidate

    parsed = urlparse(candidate)
    domain = parsed.netloc.lower()

    # Strip a leading "www." so "www.example.com" and "example.com"
    # are treated as the same domain later when comparing to email.
    if domain.startswith("www."):
        domain = domain[4:]

    # A minimally valid domain must contain at least one dot and no
    # spaces, and must not be empty.
    is_valid = bool(domain) and "." in domain and " " not in domain

    return is_valid, (domain if is_valid else None)


# ==================================================================
# STEP 3 — EMAIL STRUCTURE CHECK
# ==================================================================
def validate_email_structure(email: str):
    """
    WHAT: Checks whether the email LOOKS structurally valid
          (something@something.tld) using a simple regular
          expression. This is NOT a mailbox-existence check.
    RETURNS: (is_valid: bool, domain: str or None)
    """
    if not is_provided(email):
        return False, None

    candidate = email.strip()
    is_valid = bool(EMAIL_PATTERN.match(candidate))
    domain = candidate.split("@")[-1].lower() if is_valid else None

    return is_valid, domain


# ==================================================================
# STEP 4 — FREE EMAIL PROVIDER CHECK
# ==================================================================
def is_free_email_provider(email_domain: str) -> bool:
    """
    WHAT: Checks whether the email domain belongs to a well-known
          free/public email service.
    WHY IT'S ONLY A WEAK SIGNAL: Many genuine small businesses,
          freelancers, and early-stage startups also use free email
          providers, so this alone should never be treated as proof
          of anything.
    """
    if not email_domain:
        return False
    return email_domain.lower() in FREE_EMAIL_PROVIDERS


# ==================================================================
# STEP 5 — DOMAIN COMPARISON
# ==================================================================
def compare_domains(website_domain, email_domain) -> str:
    """
    WHAT: Compares the website domain and email domain using an
          exact string match (after both have already been
          lowercased and had a leading "www." stripped).
    RETURNS: one of "match", "mismatch", or "not_comparable"
          (the last one means we don't have two valid domains to
          compare in the first place — e.g. the website was missing
          or invalid).
    KNOWN LIMITATION (worth mentioning in a viva): this is an exact
          match only. A legitimate subdomain such as
          "careers.example.com" (email) vs. "example.com" (website)
          will be reported as a "mismatch", even though many real
          companies genuinely use a careers/recruiting subdomain.
          This is a deliberate simplicity trade-off for a
          beginner-friendly, explainable first version — exact-match
          keeps the logic easy to reason about and demonstrate,
          at the cost of some false-positive mismatches. A more
          advanced version could compare only the "registrable"
          root domain instead.
    """
    if not website_domain or not email_domain:
        return "not_comparable"
    return "match" if website_domain == email_domain else "mismatch"


# ==================================================================
# STEP 6 — SCORING
# ==================================================================
def compute_verification_score(
    name_provided: bool,
    website_provided: bool,
    website_valid: bool,
    email_provided: bool,
    email_valid: bool,
    free_email: bool,
    domain_comparison: str,
):
    """
    WHAT: Adds up risk points for each signal detected, clips the
          total to [0, 100], and returns both the score and a list
          of plain-English findings explaining every point added or
          subtracted.
    WHY ADDITIVE POINTS (not a formula/model): This keeps every part
          of the score traceable to a specific, named reason — ideal
          for explaining in a viva, unlike a black-box calculation.
    """
    score = 0
    findings = []

    if not name_provided:
        score += POINTS_NAME_MISSING
        findings.append(f"Company name was not provided (+{POINTS_NAME_MISSING} risk points).")
    else:
        findings.append("Company name was provided.")

    if not website_provided:
        score += POINTS_WEBSITE_MISSING
        findings.append(f"Company website was not provided (+{POINTS_WEBSITE_MISSING} risk points).")
    elif not website_valid:
        score += POINTS_WEBSITE_INVALID
        findings.append(
            f"Company website was provided but does not look like a valid "
            f"URL/domain (+{POINTS_WEBSITE_INVALID} risk points)."
        )
    else:
        findings.append("Company website was provided and looks structurally valid.")

    if not email_provided:
        score += POINTS_EMAIL_MISSING
        findings.append(f"Application email was not provided (+{POINTS_EMAIL_MISSING} risk points).")
    elif not email_valid:
        score += POINTS_EMAIL_INVALID
        findings.append(
            f"Application email was provided but does not look structurally "
            f"valid (+{POINTS_EMAIL_INVALID} risk points)."
        )
    else:
        findings.append("Application email was provided and looks structurally valid.")

    if email_valid and free_email:
        score += POINTS_FREE_EMAIL
        findings.append(
            f"Application email uses a free/public email provider "
            f"(+{POINTS_FREE_EMAIL} risk points). This alone is common for "
            f"genuine small businesses too, and is only a weak signal."
        )
    elif email_valid:
        findings.append("Application email does not use a common free/public email provider.")

    if domain_comparison == "mismatch":
        score += POINTS_DOMAIN_MISMATCH
        findings.append(
            f"The website domain and the email domain do not match "
            f"(+{POINTS_DOMAIN_MISMATCH} risk points). This can happen for "
            f"legitimate reasons (e.g. a separate recruiting or marketing "
            f"domain), so it is a warning signal, not proof."
        )
    elif domain_comparison == "match":
        score += POINTS_DOMAIN_MATCH_BONUS
        findings.append(
            f"The website domain and the email domain match "
            f"({POINTS_DOMAIN_MATCH_BONUS} risk points — a reassuring signal)."
        )
    else:
        findings.append(
            "Website and email domains could not be compared (one or both "
            "were missing or structurally invalid)."
        )

    score = max(0, min(100, score))
    return score, findings


def score_to_verification_level(score: float) -> str:
    """Maps the 0-100 verification risk score to a Low/Medium/High
    label, using the same cut points as the Day 4 ML risk score for
    a consistent user experience: 0-29 Low, 30-59 Medium, 60-100 High."""
    if score <= 29:
        return "Low Risk"
    elif score <= 59:
        return "Medium Risk"
    else:
        return "High Risk"


# ==================================================================
# MAIN PUBLIC FUNCTION
# ==================================================================
def verify_company(company_name: str = None, company_website: str = None,
                    application_email: str = None) -> dict:
    """
    WHAT: The main entry point for this module. Runs every check
          above on the three supplied fields and returns one
          structured result dictionary.

    ARGS:
        company_name:       free-text company name, or None/"" if unknown
        company_website:    free-text website/URL, or None/"" if unknown
        application_email:  free-text email address, or None/"" if unknown

    RETURNS: a dict with:
        company_name_provided     : bool
        website_provided          : bool
        website_valid_structure   : bool
        website_domain            : str or None
        email_provided            : bool
        email_valid_structure     : bool
        email_domain               : str or None
        is_free_email_provider    : bool
        domain_comparison          : "match" / "mismatch" / "not_comparable"
        verification_score        : int, 0-100 (higher = more warning signals)
        verification_level        : "Low Risk" / "Medium Risk" / "High Risk"
        verification_findings     : list[str], plain-English explanations
        disclaimer                : str
    """
    name_provided = is_provided(company_name)

    website_provided = is_provided(company_website)
    website_valid, website_domain = validate_website_structure(company_website)

    email_provided = is_provided(application_email)
    email_valid, email_domain = validate_email_structure(application_email)

    free_email = is_free_email_provider(email_domain) if email_valid else False

    domain_comparison = compare_domains(website_domain, email_domain)

    score, findings = compute_verification_score(
        name_provided=name_provided,
        website_provided=website_provided,
        website_valid=website_valid,
        email_provided=email_provided,
        email_valid=email_valid,
        free_email=free_email,
        domain_comparison=domain_comparison,
    )
    level = score_to_verification_level(score)

    return {
        "company_name_provided": name_provided,
        "website_provided": website_provided,
        "website_valid_structure": website_valid,
        "website_domain": website_domain,
        "email_provided": email_provided,
        "email_valid_structure": email_valid,
        "email_domain": email_domain,
        "is_free_email_provider": free_email,
        "domain_comparison": domain_comparison,
        "verification_score": score,
        "verification_level": level,
        "verification_findings": findings,
        "disclaimer": DISCLAIMER,
    }


# ==================================================================
# STANDALONE TEST BLOCK
# ==================================================================
if __name__ == "__main__":
    sample_cases = [
        {
            "label": "Case 1 — consistent, well-formed details",
            "company_name": "Brightline Analytics Inc.",
            "company_website": "https://www.brightlineanalytics.com",
            "application_email": "careers@brightlineanalytics.com",
        },
        {
            "label": "Case 2 — free email + no website",
            "company_name": "QuickHire Solutions",
            "company_website": "",
            "application_email": "quickhire.jobs@gmail.com",
        },
        {
            "label": "Case 3 — mismatched domains",
            "company_name": "NovaTech Systems",
            "company_website": "https://www.novatechsystems.com",
            "application_email": "hr@novatech-careers-hiring.net",
        },
        {
            "label": "Case 4 — nothing provided",
            "company_name": "",
            "company_website": "",
            "application_email": "",
        },
    ]

    for case in sample_cases:
        print("=" * 70)
        print(case["label"])
        print("=" * 70)
        result = verify_company(
            company_name=case["company_name"],
            company_website=case["company_website"],
            application_email=case["application_email"],
        )
        print(f"Verification score: {result['verification_score']} / 100")
        print(f"Verification level: {result['verification_level']}")
        print(f"Website domain:     {result['website_domain']}")
        print(f"Email domain:       {result['email_domain']}")
        print(f"Domain comparison:  {result['domain_comparison']}")
        print("Findings:")
        for finding in result["verification_findings"]:
            print(f"  - {finding}")
        print(f"\nDisclaimer: {result['disclaimer']}")
        print()
