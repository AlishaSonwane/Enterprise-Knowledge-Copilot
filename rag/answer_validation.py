"""Conservative cloud output-format checks, not factual entailment validation."""
import re

CITATION = re.compile(r'\[Source:\s*([^\]]+)\]')

def validate_cloud_answer(answer, allowed_sources):
    citations = CITATION.findall(answer)
    if not citations or any(c not in allowed_sources for c in citations):
        return 'cloud_invalid_citations'
    # Intentionally narrow warning patterns; absence does not prove correctness.
    if re.search(r'\b(?:guarantee\w*|ensur\w*)\b.{0,100}\b(?:correct|accurate|accuracy|verifiable|factual|hallucination)\w*\b', answer, re.I | re.S):
        return 'cloud_overstated_guarantee'
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        matches = list(CITATION.finditer(line))
        if not matches or line[matches[-1].end():].strip():
            return 'cloud_citation_placement'
        start = 0
        for match in matches:
            segment = line[start:match.start()].strip()
            # Conservative sentence boundary heuristic. Abbreviations can reject
            # valid text; never silently rewrite or append invented citations.
            if not segment or re.search(r'[.!?]\s+\S', segment):
                return 'cloud_citation_placement'
            start = match.end()
    return None
