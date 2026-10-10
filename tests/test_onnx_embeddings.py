import unittest
from unittest.mock import Mock
import numpy as np
from rag.onnx_embeddings import MiniLM

class OnnxTests(unittest.TestCase):
    def model(self):
        m=object.__new__(MiniLM)
        token=Mock(ids=[101,102,0],attention_mask=[1,1,0],type_ids=[0,0,0])
        m.tokenizer=Mock();m.tokenizer.encode_batch.return_value=[token]
        m.session=Mock();m.session.get_inputs.return_value=[Mock(name=n) for n in []]
        hidden=np.zeros((1,3,384),dtype=np.float32)
        hidden[0,0,0]=1;hidden[0,1,1]=1;hidden[0,2,:]=100
        m.session.run.return_value=[hidden]
        return m
    def test_masked_pooling_and_normalization(self):
        m=self.model();out=m.encode(['test'])
        self.assertEqual(out.shape,(1,384))
        self.assertAlmostEqual(float(out[0,0]),2**-.5,places=6)
        self.assertEqual(float(out[0,2]),0)
        self.assertAlmostEqual(float(np.linalg.norm(out[0])),1,places=6)
    def test_empty_input(self):self.assertEqual(self.model().encode([]).shape,(0,384))
    def test_invalid_batch(self):
        with self.assertRaises(ValueError):self.model().encode(['x'],batch_size=0)
    def test_nonfinite_embedding_rejected(self):
        m=self.model();m.session.run.return_value=[np.full((1,3,384),np.nan)]
        with self.assertRaises(ValueError):m.encode(['x'])
