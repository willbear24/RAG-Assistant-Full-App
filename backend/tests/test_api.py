from fastapi.testclient import TestClient
from my_rag_api import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "chromadb" in data


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "document_count" in data


def test_ask_blank_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_missing_question_field():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_no_relevant_documents():
    # A max_distance of 0 means no chunk can match, so Ollama is never called
    response = client.post("/ask", json={"question": "anything", "max_distance": 0})
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"] == "none"
    assert data["sources"] == []


def test_health_ollama_status_is_valid():
    data = client.get("/health").json()
    assert data["ollama"] in ("ok", "unavailable")


def test_wrong_method_and_unknown_route():
    assert client.get("/ask").status_code == 405
    assert client.get("/does-not-exist").status_code == 404
