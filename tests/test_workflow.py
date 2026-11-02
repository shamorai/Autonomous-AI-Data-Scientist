import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add root folder path to Python path dynamically
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

def test_profile_endpoint_invalid_file():
    response = client.post("/api/profile", files={"file": ("test.txt", b"invalid data", "text/plain")})
    assert response.status_code == 400