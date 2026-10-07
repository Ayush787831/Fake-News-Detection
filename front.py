# -*- coding: utf-8 -*-
"""
Front web entry point for Fake News Detection.
"""
from app import app

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Fake News Detection Front Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)