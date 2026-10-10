import ast,asyncio,pathlib,types,unittest
from dataclasses import dataclass

class MetadataTests(unittest.TestCase):
    def test_provider_metadata_no_secrets(self):
        root=pathlib.Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'app.py').read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='get_stats')
        function.decorator_list=[]  # Isolate route logic, not FastAPI registration.
        @dataclass
        class Metrics:total_queries:int=0
        from dataclasses import asdict
        for provider in ('groq','ollama'):
            config=types.SimpleNamespace(GENERATION_PROVIDER=provider,GROQ_MODEL='cloud-model',OLLAMA_MODEL='local-model',EMBEDDING_BACKEND='onnx',EMBEDDING_MODEL='MiniLM',CHUNK_SIZE=300,CHUNK_OVERLAP=50,TOP_K_RETRIEVAL=20,TOP_K_RERANK=5,HYBRID_ALPHA=.7,DOCUMENTS_DIR='unused',GROQ_API_KEY='secret')
            namespace={'config':config,'get_index_stats':lambda:{},'get_metrics':lambda:Metrics(),'asdict':asdict,'PARSERS':{},'os':types.SimpleNamespace(path=types.SimpleNamespace(isdir=lambda p:False))}
            exec(compile(ast.Module(body=[function],type_ignores=[]),'isolated_stats','exec'),namespace)
            result=asyncio.run(namespace['get_stats']())
            self.assertEqual(result['config']['generation_model'],'cloud-model' if provider=='groq' else 'local-model')
            self.assertNotIn('secret',str(result));self.assertNotIn('GROQ_API_KEY',str(result))
