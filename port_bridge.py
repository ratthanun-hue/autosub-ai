#!/usr/bin/env python3
"""
Port Bridge: Forwards incoming requests on port 10100 to internal Master LLM port 10200.
Enables Vast.ai console's default 'Open' button (mapped to 10100) to reach Master LLM dashboard.
"""
import http.server
import socketserver
import urllib.request
import urllib.error

PORT = 10100
TARGET = 'http://127.0.0.1:10200'

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.proxy_request('GET')

    def do_POST(self):
        self.proxy_request('POST')

    def do_HEAD(self):
        self.proxy_request('HEAD')

    def proxy_request(self, method):
        target_url = f'{TARGET}{self.path}'
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else None

        req_headers = {k: v for k, v in self.headers.items() if k.lower() not in ['host', 'content-length']}
        req = urllib.request.Request(target_url, data=post_data, headers=req_headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ['transfer-encoding', 'content-length']:
                        self.send_header(k, v)
                resp_data = resp.read()
                self.send_header('Content-Length', str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ['transfer-encoding', 'content-length']:
                    self.send_header(k, v)
            resp_data = e.read()
            self.send_header('Content-Length', str(len(resp_data)))
            self.end_headers()
            self.wfile.write(resp_data)
        except Exception as ex:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f'Bad Gateway: {ex}'.encode('utf-8'))

    def log_message(self, format, *args):
        pass # Quiet logging

if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(('0.0.0.0', PORT), ProxyHandler) as httpd:
        print(f'Port bridge listening on 10100 -> forwarding to 10200')
        httpd.serve_forever()
