import pytest
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean(text):
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    return text

def parse_json(text):
    text = clean(text)
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])

def validate_company(company):
    if not company:
        return False
    if len(company) > 100:
        return False
    if any(char in company for char in ['<', '>', '{', '}']):
        return False
    return True

def estimate_tokens(text):
    return len(text) // 4

def add_to_history(history, role, content):
    if role not in ["user", "assistant"]:
        raise ValueError(f"Invalid role: {role}")
    history.append({"role": role, "content": content})
    return history

def test_parse_json_clean():
    text = '{"company": "Google", "difficulty": "high"}'
    result = parse_json(text)
    assert result["company"] == "Google"
    assert result["difficulty"] == "high"

def test_parse_json_with_extra_text():
    text = 'Here is the JSON: {"company": "Amazon"} Hope this helps!'
    result = parse_json(text)
    assert result["company"] == "Amazon"

def test_parse_json_with_think_tags():
    text = '<think>thinking...</think>{"company": "Microsoft"}'
    result = parse_json(text)
    assert result["company"] == "Microsoft"

def test_parse_json_invalid():
    with pytest.raises(Exception):
        parse_json("this is not json at all")

def test_clean_removes_think_tags():
    text = "<think>some thinking</think>actual answer"
    result = clean(text)
    assert result == "actual answer"

def test_clean_no_think_tags():
    text = "normal text without think tags"
    result = clean(text)
    assert result == "normal text without think tags"

def test_clean_empty_string():
    result = clean("")
    assert result == ""

def test_validate_company_valid():
    assert validate_company("Google") == True
    assert validate_company("Amazon Web Services") == True
    assert validate_company("Tata Consultancy Services") == True

def test_validate_company_empty():
    assert validate_company("") == False
    assert validate_company(None) == False

def test_validate_company_too_long():
    long_name = "A" * 101
    assert validate_company(long_name) == False

def test_validate_company_injection():
    assert validate_company("<script>alert('hack')</script>") == False
    assert validate_company("{malicious}") == False

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0

def test_estimate_tokens_short():
    result = estimate_tokens("Hello world")
    assert result > 0

def test_estimate_tokens_long():
    long_text = "word " * 100
    result = estimate_tokens(long_text)
    assert result > 50

def test_add_to_history_user():
    history = []
    result = add_to_history(history, "user", "Hello")
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"

def test_add_to_history_assistant():
    history = []
    result = add_to_history(history, "assistant", "Hi there!")
    assert len(result) == 1
    assert result[0]["role"] == "assistant"

def test_add_to_history_invalid_role():
    history = []
    with pytest.raises(ValueError):
        add_to_history(history, "system", "Invalid")

def test_add_to_history_multiple():
    history = []
    add_to_history(history, "user", "Hello")
    add_to_history(history, "assistant", "Hi!")
    add_to_history(history, "user", "How are you?")
    assert len(history) == 3
