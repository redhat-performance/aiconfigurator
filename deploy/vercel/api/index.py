# Vercel Python Function entrypoint.
# Imports the FastAPI app so Vercel serves it as a serverless function.

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tools.api_service.app import app
