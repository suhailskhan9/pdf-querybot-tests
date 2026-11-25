import pytest

from helpers import random_uuid

pytestmark = pytest.mark.api


@pytest.mark.slow
def test_delete_session_then_ask_returns_404(api, fresh_session):
    """Protects: deleting a session really removes it. DELETE returns 200 with the
    documented message, and a later /ask on that id gives 404 (no model call is made)."""
    resp = api.delete(f"/session/{fresh_session}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": "Session deleted"}

    ask_resp = api.ask("Who founded the Zorblat project?", fresh_session)
    assert ask_resp.status_code == 404, ask_resp.text
    assert isinstance(ask_resp.json().get("detail"), str)


def test_delete_invalid_session_id_returns_400(api):
    """Protects: a non-UUID id in the DELETE path is rejected with 400."""
    resp = api.delete("/session/not-a-uuid")
    assert resp.status_code == 400, resp.text
    assert isinstance(resp.json().get("detail"), str)


def test_delete_unknown_session_is_idempotent(api):
    """Protects: deleting a valid but non-existent session still returns 200, so cleanup
    and retries never fail. Costs nothing."""
    resp = api.delete(f"/session/{random_uuid()}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": "Session deleted"}