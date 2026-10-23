"""Serve the saved simulation locally; has no robot connection or write API."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
from urllib.parse import quote
import argparse, subprocess, sys, json, urllib.request, webbrowser, time
from .data import ROOT

PORT=8774
URL=f'http://127.0.0.1:{PORT}/cbf_equipment_20261002/'+quote('医疗设备与CBF仿真.html')

def serve():
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
        def end_headers(self):self.send_header('Cache-Control','no-cache');super().end_headers()
        def do_GET(self):
            if self.path=='/cbf-health':
                raw=json.dumps({'service':'NC-CBF-equipment','root':str(ROOT)}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
            return super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',PORT),partial(Handler,directory=str(ROOT/'outputs')))
    print(URL,flush=True);server.serve_forever()

def start():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def health():
        try:
            with opener.open(f'http://127.0.0.1:{PORT}/cbf-health',timeout=.6) as r:return json.load(r).get('root')==str(ROOT)
        except (OSError,ValueError):return False
    if not health():
        log=(ROOT/'outputs/cbf_equipment_20261002/server.log').open('ab')
        subprocess.Popen([sys.executable,'-m','ncsim.cbf_server'],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        log.close()
        for _ in range(50):
            if health():break
            time.sleep(.2)
        else:raise RuntimeError('Local server could not start; see server.log')
    return URL

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--open',action='store_true');p.add_argument('--start',action='store_true');a=p.parse_args()
    if a.open:webbrowser.open(start())
    elif a.start:print(start())
    else:serve()
