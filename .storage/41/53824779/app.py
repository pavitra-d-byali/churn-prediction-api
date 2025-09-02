from app import create_app
import os
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

app = create_app()

if __name__ == '__main__':
    # Find an available port dynamically
    port = find_free_port()
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)