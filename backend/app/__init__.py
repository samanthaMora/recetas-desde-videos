import os

import certifi

# Algunas instalaciones de Python en macOS no traen certificados raíz, y las descargas
# con urllib (p. ej. los modelos de EasyOCR) fallan con CERTIFICATE_VERIFY_FAILED.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
