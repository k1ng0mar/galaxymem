"""Tests for secret detection/redaction at retain time."""

from galaxymem.redact import find_secrets, redact_secrets


def test_detects_openai_style_key():
    text = "my key is sk-proj-abc123def456GHI789jklMNO"
    assert len(find_secrets(text)) == 1


def test_detects_bearer_jwt():
    hits = find_secrets(
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig"
    )
    assert any(h.kind == "jwt" for h in hits)


def test_detects_aws_access_key():
    assert find_secrets("AKIAIOSFODNN7EXAMPLE")


def test_detects_github_token():
    assert find_secrets("token ghp_abcdefghijklmnopqrstuvwxyz012345")


def test_detects_password_assignment():
    hits = find_secrets("password=hunter2secret")
    assert any(h.kind == "credential_assignment" for h in hits)


def test_detects_private_key_block():
    assert find_secrets("-----BEGIN RSA PRIVATE KEY-----")


def test_clean_text_has_no_hits():
    assert find_secrets("the api runs on port 8010") == []
    # 'sk-' alone must not false-positive on normal words
    assert find_secrets("task risk assessment") == []


def test_redact_preserves_surroundings():
    text = "use key sk-proj-abc123def456GHI789jklMNO tomorrow"
    out = redact_secrets(text)
    assert "[REDACTED]" in out
    assert "sk-proj" not in out
    assert out.startswith("use key ") and out.endswith(" tomorrow")


def test_redact_idempotent():
    text = "password=hunter2secret and AKIAIOSFODNN7EXAMPLE"
    once = redact_secrets(text)
    twice = redact_secrets(once)
    assert once == twice


def test_empty_and_none_safe():
    assert redact_secrets("") == ""
    assert find_secrets("") == []
