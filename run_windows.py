"""Local Windows cloud/ONNX launcher. No installs, downloads or global settings."""
import argparse,ctypes,datetime,hashlib,json,os,pathlib,shutil,signal,socket,subprocess,sys,time,urllib.request

ROOT=pathlib.Path(__file__).resolve().parent
DLL_HASHES={'msvcp140.dll':'7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca',
            'msvcp140_1.dll':'206c931bf90fdad8816de3b5e2ef80b2bcaa9406c89ecc05fe6fddffe251e982'}
ENV_KEYS={'GENERATION_PROVIDER','GROQ_FREE_PLAN_CONFIRMED','GROQ_API_KEY','GROQ_MAX_COMPLETION_TOKENS'}

def read_profile(root):
    profile=json.loads((root/'.launcher.json').read_text(encoding='utf-8-sig'))
    for key in ('python','model','tokenizer'):
        if not isinstance(profile.get(key),str) or not pathlib.Path(profile[key]).is_file():
            raise ValueError('Local launcher path missing: '+key)
    return profile

def cloud_env(root,profile):
    env=os.environ.copy()
    for line in (root/'.env').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        key,sep,value=line.partition('=')
        if sep and key.strip() in ENV_KEYS:env[key.strip()]=value.strip()
    if env.get('GENERATION_PROVIDER')!='groq' or env.get('GROQ_FREE_PLAN_CONFIRMED')!='1' or not env.get('GROQ_API_KEY'):
        raise ValueError('Local Groq Free configuration incomplete; key not displayed')
    env.update(EMBEDDING_BACKEND='onnx',ONNX_MODEL_PATH=profile['model'],ONNX_TOKENIZER_PATH=profile['tokenizer'],
        PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',ANONYMIZED_TELEMETRY='False')
    return env

def available_ram():
    if os.name!='nt':raise RuntimeError('This launcher is Windows only')
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ['total','avail','page','availpage','virtual','availvirtual','extra']]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError('Memory check failed')
    return m.avail/2**30

def serve(profile,port,out,smoke):
    # Optional workaround is process-local, exact verified files only.
    handle=None
    if profile.get('dll_directory'):
        directory=pathlib.Path(profile['dll_directory'])
        for name,expected in DLL_HASHES.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Native runtime identity changed')
        handle=os.add_dll_directory(str(directory))
    sys.path.insert(0,str(ROOT))
    import config
    if smoke:
        config.DATA_DIR=str(out/'smoke-data');config.CHROMA_DIR=str(out/'smoke-data/chroma_db')
    import uvicorn
    server=uvicorn.Server(uvicorn.Config('app:app',host='127.0.0.1',port=port,log_level='info'))
    signal.signal(signal.SIGBREAK,lambda sig,frame:setattr(server,'should_exit',True))
    server.run()

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--serve',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--port',type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument('--out',default='',help=argparse.SUPPRESS)
    parser.add_argument('--max-seconds',type=int,default=3600)
    args=parser.parse_args(argv)
    if not 1<=args.max_seconds<=14400:raise ValueError('Runtime must be between 1 and 14400 seconds')
    profile=read_profile(ROOT)
    if args.serve:
        if not 1<=args.port<=65535:raise ValueError('Invalid loopback port')
        if available_ram()<1.5:raise RuntimeError('STOP: child startup RAM below 1.5 GiB')
        serve(profile,args.port,pathlib.Path(args.out),args.smoke_test);return 0
    if available_ram()<1.5:raise RuntimeError('STOP: available RAM below 1.5 GiB; application not started')
    env=cloud_env(ROOT,profile)
    out=ROOT/'data/launcher'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True,exist_ok=False)
    if args.smoke_test:
        shutil.copytree(ROOT/'data/chroma_db',out/'smoke-data/chroma_db')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    result={'url':url,'smoke_test':args.smoke_test,'success':False};samples=[];child=None
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started=time.monotonic()
    try:
        with (out/'server.log').open('w',encoding='utf-8') as log:
            command=[profile['python'],'-I','-B','-X','utf8',str(ROOT/'run_windows.py'),'--serve','--port',str(port),'--out',str(out)]
            if args.smoke_test:command.append('--smoke-test')
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            ready=False
            while child.poll() is None:
                elapsed=time.monotonic()-started;ram=available_ram()
                samples.append({'seconds':round(elapsed,1),'available_gib':ram})
                if ram<0.5:raise RuntimeError('STOP: severe memory pressure')
                if not ready:
                    if elapsed>90:raise RuntimeError('Startup timeout; inspect server.log')
                    try:
                        with opener.open(url+'/openapi.json',timeout=1) as response:ready=response.status==200
                    except OSError:pass
                    if ready:
                        print('Ready: '+url,flush=True)
                        print('Stop with Ctrl+C. Logs: '+str(out),flush=True)
                        result['openapi_status']=200
                        if args.smoke_test:
                            for route in ('/','/api/logs'):
                                with opener.open(url+route,timeout=5) as response:result[route]=response.status;response.read()
                            result['success']=True;break
                if elapsed>=args.max_seconds:
                    result['success']=ready;result['reason']='Bounded session finished';break
                time.sleep(1)
            if not ready:raise RuntimeError('Server exited before readiness')
            if child.poll() is not None and not args.smoke_test:raise RuntimeError('Server exited unexpectedly')
    except KeyboardInterrupt:
        result['success']=result.get('openapi_status')==200;result['reason']='Stopped by user'
    except Exception as error:
        result['error']=str(error);print(str(error),flush=True)
    finally:
        if child and child.poll() is None:
            child.send_signal(signal.CTRL_BREAK_EVENT)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=5);result['forced_stop']=True
        result['server_exited']=child is None or child.poll() is not None
        with socket.socket() as sock:result['port_released']=sock.connect_ex(('127.0.0.1',port))!=0
        (out/'memory.json').write_text(json.dumps(samples),encoding='utf-8')
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Results: '+str(out),flush=True)
    return 0 if result['success'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:print(str(error));sys.exit(1)
