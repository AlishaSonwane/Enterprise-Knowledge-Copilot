"""Pricing support regressions; unsupported pricing must abstain.

Recorded 8C evidence/ranking scores are replayed using synthetic unit vectors.
This tests the real retrieval/guardrail branches, not embedding calibration.
No database, model, HTTP service or generation is used.
"""
import math
import unittest
from unittest.mock import patch
import numpy as np
from test_regressions import retrieval, guardrails

QUERY = 'What is the exact monthly subscription price for our company private RAG service?'
RECORDED = [
    ('rag_system_guide.md', 0.3363784796172692, 0.01639344262295082),
    ('vector_databases.md', 0.08191365987952207, 0.01597782258064516),
    ('rag_system_guide.md', 0.053972911698050775, 0.015873015873015872),
]


class PricingSupportRegression(unittest.TestCase):
    def replay(self, reranking):
        candidates = [dict(chunk_id=str(i), text='General RAG/vector design guide; no company price.',
                           metadata={'source': source}, score=rank,
                           embedding=np.array([cosine, math.sqrt(1-cosine*cosine)]))
                      for i, (source, cosine, rank) in enumerate(RECORDED)]
        with patch.object(retrieval, 'embed_query', return_value=np.array([1., 0.])), \
                patch.object(retrieval, 'hybrid_search', return_value=candidates):
            # top_k=1 exercises actual reranking with multiple candidates.
            result = retrieval.retrieve(QUERY, top_k=1, use_reranking=reranking)
        self.assertAlmostEqual(result.top_score, RECORDED[0][1])
        self.assertEqual(result.chunks[0]['metadata']['source'], 'rag_system_guide.md')
        self.assertEqual('combined_score' in result.chunks[0], reranking)
        return guardrails.check_safety(QUERY, result)

    def test_recorded_false_acceptance_now_blocked(self):
        for mode in (True, False):
            with self.subTest(reranking=mode):
                self.assertFalse(self.replay(mode).is_safe)
                self.assertEqual(self.replay(mode).reason, "insufficient_evidence_support")

    def test_company_pricing_must_abstain_reranking_enabled(self):
        self.assertFalse(self.replay(True).is_safe,
                         'Topical similarity must not authorize company pricing')

    def test_company_pricing_must_abstain_reranking_disabled(self):
        self.assertFalse(self.replay(False).is_safe,
                         'Topical similarity must not authorize company pricing')


if __name__ == '__main__':
    unittest.main()
