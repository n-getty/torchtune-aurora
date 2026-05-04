#!/usr/bin/env python3
"""
Minimal async round-robin HTTP proxy for multi-instance vLLM benchmarking.
Listens on LISTEN_PORT, distributes requests round-robin to BACKENDS.
Usage: python proxy_round_robin.py <listen_port> <backend1> <backend2> ...
       python proxy_round_robin.py 8000 http://localhost:8001 http://localhost:8002 http://localhost:8003
"""
import asyncio
import sys
import itertools
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

backends = []
_cycle = None

def next_backend():
    return next(_cycle)

class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence per-request logging

    def do_request(self):
        backend = next_backend()
        url = backend + self.path
        body = None
        length = int(self.headers.get('Content-Length', 0))
        if length:
            body = self.rfile.read(length)

        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ('host', 'content-length', 'transfer-encoding')}
        if body:
            headers['Content-Length'] = str(len(body))

        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ('transfer-encoding',):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())

    do_GET  = do_request
    do_POST = do_request

if __name__ == '__main__':
    listen_port = int(sys.argv[1])
    backends = sys.argv[2:]
    _cycle = itertools.cycle(backends)
    print(f"Round-robin proxy on :{listen_port} → {backends}", flush=True)
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(('localhost', listen_port), ProxyHandler)
    server.serve_forever()
