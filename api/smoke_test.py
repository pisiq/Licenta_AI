from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.main import app


if __name__ == "__main__":
    client = TestClient(app)
    response = client.get("/v1/health")
    print("Status:", response.status_code)
    print("Body:", response.json())

