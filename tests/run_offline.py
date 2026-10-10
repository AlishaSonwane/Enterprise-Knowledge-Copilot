import pathlib,sys,types,unittest,ast
from unittest.mock import Mock,patch
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));sys.path.insert(0,str(root/'tests'))
for p in (root/'config.py',root/'rag/generation.py',root/'rag/cloud_generation.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
blocked=[]
class NoModels:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','chromadb','onnxruntime','sentence_transformers'):
   blocked.append(name);raise ImportError('Offline native import prohibited: '+name)
sys.meta_path.insert(0,NoModels())
e=types.ModuleType('rag.embeddings');e.embed_query=Mock()
i=types.ModuleType('rag.indexing');i.hybrid_search=Mock();i.vector_search=Mock()
with patch.dict(sys.modules,{'rag.embeddings':e,'rag.indexing':i}):
 suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*.py')
 result=unittest.TextTestRunner(verbosity=2).run(suite)
print('Native import attempts:',blocked)
sys.exit(0 if result.wasSuccessful() and not blocked else 1)
