"""macOS Virtual Filesystem Mount Provider.

Enables mounting .lmz models directly onto macOS without disk extraction.
Uses macOS built-in mount_webdav loopback with zero external dependencies.
"""

from __future__ import annotations

import http.server
import os
import subprocess
import sys
import threading
import urllib.parse
from xml.sax.saxutils import escape


class MacWebDAVHandler(http.server.BaseHTTPRequestHandler):
    """WebDAV request handler serving files and directories from ModelFS."""

    def log_message(self, format, *args):
        pass  # Suppress request logging

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("DAV", "1, 2")
        self.send_header("MS-Author-Via", "DAV")
        self.send_header("Allow", "OPTIONS, GET, HEAD, PROPFIND")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _find_node(self, path: str):
        path = urllib.parse.unquote(path).strip("/")
        if not path:
            return self.server.fs.tree.root, ""
        parts = [p for p in path.split("/") if p and p != "."]
        node = self.server.fs.tree.root
        for p in parts:
            if not node.is_dir or p not in node.children:
                return None, path
            node = node.children[p]
        return node, path

    def do_PROPFIND(self):
        node, clean_path = self._find_node(self.path)
        if node is None:
            self.send_response(404)
            self.end_headers()
            return

        depth = self.headers.get("Depth", "1")
        items = [(node, clean_path)]
        if node.is_dir and depth != "0":
            for name, child in sorted(node.children.items()):
                child_path = f"{clean_path}/{name}".strip("/")
                items.append((child, child_path))

        res = [b"<?xml version=\"1.0\" encoding=\"utf-8\" ?>\n<D:multistatus xmlns:D=\"DAV:\">\n"]
        for n, p in items:
            href = "/" + p if p else "/"
            if n.is_dir and not href.endswith("/"):
                href += "/"
            res.append(f"""  <D:response>
    <D:href>{escape(href)}</D:href>
    <D:propstat>
      <D:prop>
        <D:displayname>{escape(n.name)}</D:displayname>
        <D:resourcetype>{"<D:collection/>" if n.is_dir else ""}</D:resourcetype>
        {"<D:getcontentlength>" + str(n.size) + "</D:getcontentlength>" if not n.is_dir else ""}
        <D:getcontenttype>{"httpd/unix-directory" if n.is_dir else "application/octet-stream"}</D:getcontenttype>
      </D:prop>
      <D:status>HTTP/1.1 200 OK</D:status>
    </D:propstat>
  </D:response>\n""".encode("utf-8"))
        res.append(b"</D:multistatus>")
        body = b"".join(res)
        self.send_response(207)
        self.send_header("Content-Type", "application/xml; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        node, _ = self._find_node(self.path)
        if node is None or node.is_dir:
            self.send_response(404 if node is None else 200)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(node.size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        node, _ = self._find_node(self.path)
        if node is None or node.is_dir:
            self.send_response(404)
            self.end_headers()
            return

        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            parts = range_header[6:].split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else node.size - 1
            if end >= node.size:
                end = node.size - 1
            length = end - start + 1
            data = self.server.fs.read(node.source, start, length)
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Range", f"bytes {start}-{end}/{node.size}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            data = self.server.fs.read(node.source, 0, node.size)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class MacMountServer:
    """Manages the mount on Darwin systems using mount_webdav."""

    def __init__(self, fs, point: str):
        self.fs = fs
        self.point = os.path.abspath(point)
        self._httpd = None
        self._thread = None
        self._stop = threading.Event()
        self.mounted = False

    def mount(self) -> None:
        os.makedirs(self.point, exist_ok=True)
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MacWebDAVHandler)
        self._httpd.fs = self.fs
        port = self._httpd.server_address[1]

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="lmz-mac-mount")
        self._thread.start()

        # Mount via macOS mount_webdav -s
        url = f"http://127.0.0.1:{port}/"
        res = subprocess.run(["mount_webdav", "-s", url, self.point], capture_output=True, text=True)
        if res.returncode != 0:
            self.close()
            raise OSError(f"mount_webdav failed ({res.returncode}): {res.stderr.strip()}")
        self.mounted = True

    def serve(self) -> None:
        if not self.mounted:
            self.mount()
        try:
            while not self._stop.is_set():
                self._stop.wait(0.25)
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        if self.mounted:
            try:
                subprocess.run(["umount", self.point], check=False, capture_output=True)
            except Exception:
                pass
            self.mounted = False
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        self.fs.close()
