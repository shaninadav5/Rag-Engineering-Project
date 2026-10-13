import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import UnsupportedFileTypeError
from app.main import create_app


class QuestionBody(BaseModel):
    question: str = Field(min_length=1)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    app = create_app(settings)

    @app.get("/raise-app-error")
    def raise_app_error() -> None:
        raise UnsupportedFileTypeError(
            "Unsupported file type '.docx'.", {"supported_types": [".pdf", ".txt", ".md"]}
        )

    @app.post("/validate")
    def validate(body: QuestionBody) -> dict[str, str]:
        return {"question": body.question}

    @app.get("/crash")
    def crash() -> None:
        raise RuntimeError("secret internal detail")

    return app


def test_app_error_uses_error_envelope(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.get("/raise-app-error")

    assert response.status_code == 415
    assert response.json() == {
        "error": {
            "code": "UNSUPPORTED_FILE_TYPE",
            "message": "Unsupported file type '.docx'.",
            "details": {"supported_types": [".pdf", ".txt", ".md"]},
        }
    }


def test_validation_error_uses_error_envelope(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.post("/validate", json={"question": ""})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert [e["field"] for e in error["details"]["errors"]] == ["body.question"]


def test_unknown_route_uses_error_envelope(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "HTTP_ERROR"


def test_method_not_allowed_keeps_allow_header(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.delete("/health")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "HTTP_ERROR"
    assert "GET" in response.headers["allow"]


def test_unexpected_error_hides_internal_details(app: FastAPI) -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/crash")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "secret internal detail" not in response.text
