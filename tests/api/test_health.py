import pytest

pytestmark = pytest.mark.api


def test_health_returns_ok(api):
    """Protects: the backend is reachable and reports healthy with the documented body."""
    resp = api.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}