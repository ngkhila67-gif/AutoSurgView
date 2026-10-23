"""Read-only local playback server; binds only to loopback."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,parse_qs,quote
import json,subprocess,sys,time,urllib.request,webbrowser
from .data import ROOT
from .manuscript_viewer import PlaybackStore


def serve(out,port=8770):
    out=Path(out).resolve();store=PlaybackStore(out)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'outputs'),**kwargs)
        def log_message(self,*args):pass
        def send_bytes(self,raw,ctype):
            self.send_response(200);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            req=urlparse(self.path)
            if req.path=='/api/health':
                return self.send_bytes(json.dumps(dict(service='manuscript-playback-v1',directory=str(out))).encode(),'application/json')
            if req.path=='/api/case':
                try:
                    q=parse_qs(req.query);i=int(q['scene'][0]);j=int(q['start'][0]);case=store.case(i,j)
                    return self.send_bytes(json.dumps(case,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'),'application/json; charset=utf-8')
                except (KeyError,ValueError):return self.send_error(400,'Invalid scene/start')
                except Exception as e:
                    print(type(e).__name__,str(e),flush=True);return self.send_error(500,'Playback verification failed; see local log')
            assets={'/assets/robot-viewer.bundle.js':(ROOT/'viewer/robot-viewer.bundle.js','text/javascript'),
                    '/assets/robot_assets.json':(out/'robot_assets.json','application/json'),
                    '/assets/paper12.png':(ROOT/'data/paper_figure12.png','image/png')}
            if req.path in assets:
                p,kind=assets[req.path];return self.send_bytes(p.read_bytes(),kind)
            return super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Ready: http://127.0.0.1:{port}/{out.name}/'+quote('正文对齐仿真.html'),flush=True)
    server.serve_forever()


def open_latest():
    latest=json.loads((ROOT/'outputs/latest_manuscript.json').read_text(encoding='utf-8'));out=Path(latest['directory']).resolve();port=None
    local=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for candidate in range(8770,8780):
        try:
            with local.open(f'http://127.0.0.1:{candidate}/api/health',timeout=.4) as response:state=json.load(response)
            if state.get('service')=='manuscript-playback-v1' and state.get('directory')==str(out):port=candidate;break
        except urllib.error.HTTPError:continue
        except (OSError,ValueError):
            import socket
            with socket.socket() as sock:
                try:sock.bind(('127.0.0.1',candidate))
                except OSError:continue
            log=(out/'playback_server.log').open('ab')
            subprocess.Popen([sys.executable,'-m','ncsim.manuscript_server','--output',str(out),'--port',str(candidate)],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            log.close();port=candidate
            for _ in range(100):
                try:
                    with local.open(f'http://127.0.0.1:{port}/api/health',timeout=.5) as response:state=json.load(response)
                    if state.get('directory')==str(out):break
                except (OSError,ValueError):pass
                time.sleep(.2)
            else:raise RuntimeError('Playback server did not start; see playback_server.log')
            break
    if port is None:raise RuntimeError('No free local playback port')
    url=f'http://127.0.0.1:{port}/{out.name}/'+quote('正文对齐仿真.html');webbrowser.open(url);print(url)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--port',type=int,default=8770);p.add_argument('--open',action='store_true');a=p.parse_args()
    if a.open:open_latest()
    elif a.output:serve(a.output,a.port)
    else:p.error('Specify --output or --open')
