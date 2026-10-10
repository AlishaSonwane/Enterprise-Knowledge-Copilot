import json,pathlib,tempfile,unittest,os
from unittest.mock import patch
import run_windows as launcher

class LauncherTests(unittest.TestCase):
    def test_missing_profile_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);(root/'.launcher.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'python'):launcher.read_profile(root)
    def test_env_whitelist_and_offline_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            (root/'.env').write_text('GENERATION_PROVIDER=groq\nGROQ_FREE_PLAN_CONFIRMED=1\nGROQ_API_KEY=test-only\nPATH=malicious\n')
            profile={'model':'model.onnx','tokenizer':'tokenizer.json'}
            with patch.dict(os.environ,{'PATH':'original'},clear=True):env=launcher.cloud_env(root,profile)
            self.assertEqual(env['PATH'],'original');self.assertEqual(env['EMBEDDING_BACKEND'],'onnx')
            self.assertEqual(env['HF_HUB_OFFLINE'],'1');self.assertEqual(env['GROQ_API_KEY'],'test-only')
    def test_no_key_no_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);(root/'.env').write_text('GENERATION_PROVIDER=groq\nGROQ_FREE_PLAN_CONFIRMED=1\n')
            with patch.dict(os.environ,{},clear=True):
                with self.assertRaises(ValueError):launcher.cloud_env(root,{'model':'m','tokenizer':'t'})
    def test_low_ram_never_launches_child(self):
        with patch.object(launcher,'read_profile',return_value={}),patch.object(launcher,'available_ram',return_value=1.49),patch.object(launcher.subprocess,'Popen') as child:
            with self.assertRaisesRegex(RuntimeError,'below 1.5'):launcher.main([])
            child.assert_not_called()
    def test_unbounded_runtime_rejected(self):
        with self.assertRaisesRegex(ValueError,'Runtime'):launcher.main(['--max-seconds','0'])
