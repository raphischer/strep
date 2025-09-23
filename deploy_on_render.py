import threading
import time
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler

from main import DATABASES, load_database, scale_and_rate, Visualization

# --- Temporary "Loading" server ---
class LoadingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>STREP is starting up... loading databases. Please wait.</h2>")

def run_loading_server(httpd):
    httpd.serve_forever()

# start temporary server
print('starting temporary server')
httpd = HTTPServer(("0.0.0.0", 10000), LoadingHandler)
t = threading.Thread(target=run_loading_server, args=(httpd,))
t.start()

# load databases (will take some minutes)
databases = {}
for name, fname in DATABASES.items():
    print('LOADING', fname)
    database, meta = load_database(fname)
    databases[name] = scale_and_rate(database, meta)

# stop temporary server
print('stopping temporary server')
httpd.shutdown()
t.join()
httpd.server_close()

# launch app
print('starting STREP')
app = Visualization(databases)
server = app.server
app.run(debug=False, host='0.0.0.0', port=10000)
