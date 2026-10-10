"""Offline cloud regressions: no credentials, native model loading or live HTTP."""
import asyncio, os, unittest
from unittest.mock import AsyncMock, Mock, patch
import httpx
import config
from rag import cloud_generation as cloud
from test_regressions import generation, retrieval

class CloudTests(unittest.TestCase):
    def invoke(self,status=200,body=None,error=None):
        response=Mock(status_code=status)
        response.json.return_value=body or {'choices':[{'finish_reason':'stop','message':{'content':'RAG uses evidence. [Source: rag_system_guide.md]'}}]}
        client=Mock();client.post=AsyncMock(return_value=response,side_effect=error)
        manager=Mock();manager.__aenter__=AsyncMock(return_value=client);manager.__aexit__=AsyncMock(return_value=False)
        with patch.dict(os.environ,{'GROQ_FREE_PLAN_CONFIRMED':'1','GROQ_API_KEY':'test-secret'}),patch.object(cloud.httpx,'AsyncClient',return_value=manager):
            try:value=asyncio.run(cloud.generate_cloud_answer([{'role':'user','content':'fixture'}]))
            except cloud.CloudGenerationError as e:value=e.reason
        return value,client
    def test_payload_and_no_retry(self):
        value,client=self.invoke();self.assertIn('[Source:',value)
        client.post.assert_awaited_once()
        args=client.post.call_args
        self.assertEqual(args.args[0],'https://api.groq.com/openai/v1/chat/completions')
        self.assertEqual(args.kwargs['json']['model'],'openai/gpt-oss-20b')
        self.assertFalse(args.kwargs['json']['stream'])
    def test_rate_limit(self):self.assertEqual(self.invoke(429)[0],'cloud_rate_limited')
    def test_auth_failure(self):self.assertEqual(self.invoke(401)[0],'cloud_auth_error')
    def test_http_failure(self):self.assertEqual(self.invoke(500)[0],'cloud_http_error')
    def test_redirect_not_accepted(self):self.assertEqual(self.invoke(302)[0],'cloud_http_error')
    def test_timeout(self):self.assertEqual(self.invoke(error=httpx.ReadTimeout('test-secret'))[0],'cloud_timeout')
    def test_network_failure(self):self.assertEqual(self.invoke(error=httpx.ConnectError('test-secret'))[0],'cloud_unavailable')
    def test_empty_content(self):self.assertEqual(self.invoke(body={'choices':[{'finish_reason':'stop','message':{'content':''}}]})[0],'cloud_empty_answer')
    def test_incomplete_content(self):self.assertEqual(self.invoke(body={'choices':[{'finish_reason':'length','message':{'content':'cut'}}]})[0],'cloud_incomplete_answer')
    def test_malformed_content(self):self.assertEqual(self.invoke(body={'choices':[]})[0],'cloud_invalid_response')
    def test_missing_key_no_http(self):
        with patch.dict(os.environ,{'GROQ_FREE_PLAN_CONFIRMED':'1','GROQ_API_KEY':''}),patch.object(cloud.httpx,'AsyncClient') as factory:
            with self.assertRaisesRegex(cloud.CloudGenerationError,'cloud_key_missing'):asyncio.run(cloud.generate_cloud_answer([]))
            factory.assert_not_called()
    def test_unconfirmed_plan_no_http(self):
        with patch.dict(os.environ,{'GROQ_FREE_PLAN_CONFIRMED':'0'}),patch.object(cloud.httpx,'AsyncClient') as factory:
            with self.assertRaisesRegex(cloud.CloudGenerationError,'cloud_free_plan_unconfirmed'):asyncio.run(cloud.generate_cloud_answer([]))
            factory.assert_not_called()
    def result(self):
        c={'chunk_id':'fixture','text':'RAG uses retrieved evidence.','evidence_score':.8,'metadata':{'source':'rag_system_guide.md','chunk_index':0}}
        return retrieval.RetrievalResult([c],'What is RAG?','What is RAG?',[.8],.8,True)
    def test_integrated_answer_original_instructions(self):
        with patch.object(config,'GENERATION_PROVIDER','groq'),patch.object(cloud,'generate_cloud_answer',new=AsyncMock(return_value='RAG uses retrieved evidence. [Source: rag_system_guide.md]')) as remote,patch.object(generation,'get_ollama_client') as local:
            r=asyncio.run(generation.generate_answer('What is RAG?',self.result()))
            self.assertEqual(r['status'],'success');self.assertEqual(r['citations'],['rag_system_guide.md']);local.assert_not_called()
            self.assertEqual(remote.call_args.args[0][0]['content'],generation.SYSTEM_PROMPT)
    def test_guardrail_before_cloud(self):
        with patch.object(config,'GENERATION_PROVIDER','groq'),patch.object(cloud,'generate_cloud_answer',new=AsyncMock()) as remote:
            r=asyncio.run(generation.generate_answer("What is our company's RAG subscription price?",self.result()))
            self.assertEqual(r['reason'],'insufficient_evidence_support');remote.assert_not_called()
    def test_cloud_failure_no_local_fallback(self):
        with patch.object(config,'GENERATION_PROVIDER','groq'),patch.object(cloud,'generate_cloud_answer',new=AsyncMock(side_effect=cloud.CloudGenerationError('cloud_rate_limited'))),patch.object(generation,'get_ollama_client') as local:
            r=asyncio.run(generation.generate_answer('What is RAG?',self.result()))
            self.assertEqual(r['reason'],'cloud_rate_limited');local.assert_not_called()
    def test_invalid_citation(self):
        with patch.object(config,'GENERATION_PROVIDER','groq'),patch.object(cloud,'generate_cloud_answer',new=AsyncMock(return_value='Answer. [Source: invented.md]')):
            r=asyncio.run(generation.generate_answer('What is RAG?',self.result()))
            self.assertEqual(r['reason'],'cloud_invalid_citations')
