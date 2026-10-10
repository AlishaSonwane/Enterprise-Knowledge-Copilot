"""Conservative scope restriction, not general factual entailment validation.

There is currently no trusted company/service identity or independently vetted
pricing evidence in this application. Retrieved text/metadata cannot establish
its own authority. Company-specific subscription pricing therefore fails closed.
Enabling supported pricing requires a separately reviewed provenance mechanism;
do not replace this restriction with keywords, currency matching or similarity.
"""
import re

_PRICE = re.compile(
    r"\b(?:price|pricing|cost|costs|fee|fees|charge|charges|charged|pay|paying|"
    r"payment|payments|bill|billing|rate|rates|subscription|tariff)\b", re.I)
_COMPANY = re.compile(
    r"\b(?:company|companies|organization|organisation|business|enterprise|"
    r"employer|corporate)\b|\b(?:our|my|ours|we|us)\b|\b[\w-]+['’]s\b", re.I)
_SERVICE = re.compile(
    r"\b(?:subscription|service|plan|platform|product|software|rag|copilot|"
    r"saas|licen[cs]e|tier|package|monthly|annual|yearly|per month)\b", re.I)


def requires_company_pricing_evidence(query: str) -> bool:
    """Classify scope conservatively; matching words do NOT establish support."""
    return bool(_PRICE.search(query) and _COMPANY.search(query)
                and _SERVICE.search(query))


def has_company_pricing_support(query: str, chunks: list[dict]) -> bool:
    """Fail closed until trusted scoped evidence can actually be verified.

No current input can verify company/service identity, authoritative provenance,
amount, currency and billing period. In particular, a retrieved `trusted=True`
flag, source filename, price-like sentence or high cosine is insufficient.
The chunks are intentionally not promoted into an authority registry.
"""
    return False


def check_evidence_support(query: str, chunks: list[dict]) -> bool:
    if not requires_company_pricing_evidence(query):
        return True
    return has_company_pricing_support(query, chunks)
