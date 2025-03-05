# シンプルなHTTPサーバー（Python 3）
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket

def get_ip():
    """ローカルIPアドレスを取得する関数"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # このIPは実際には接続しません
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# サーバーの設定
host = '127.0.0.1'  # localhost
port = 8000
server_address = (host, port)

# HTTPサーバーを起動
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
print(f"サーバーを起動しました: http://{host}:{port}/")
print(f"停止するには Ctrl+C を押してください")

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nサーバーを停止しています...")
    httpd.server_close()
    print("サーバーを停止しました")