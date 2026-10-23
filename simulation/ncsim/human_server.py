"""Open the independent human-aware replay on the existing read-only server."""
import webbrowser
from urllib.parse import quote
from .cbf_server import start,PORT
URL=f'http://127.0.0.1:{PORT}/reference_equipment_20261012/'+quote('人体与CBF安全仿真.html')
if __name__=='__main__':
    start();webbrowser.open(URL)
