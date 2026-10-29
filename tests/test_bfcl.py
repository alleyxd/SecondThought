from toolcall.bfcl import check_call, convert_schema, safe_name, score, validate

TRUTH = {"get_weather": {"city": ["Calgary", "calgary, ab"], "unit": ["celsius", ""]}}


def test_safe_name():
    assert safe_name("math.factorial") == "math_factorial"


def test_check_call_accepts_any_listed_value_and_optional_args():
    names = {"get_weather": "get_weather"}
    assert check_call("get_weather", {"city": "Calgary"}, TRUTH, names)
    assert check_call("get_weather", {"city": "Calgary, AB", "unit": "celsius"}, TRUTH, names)
    assert not check_call("get_weather", {"city": "Toronto"}, TRUTH, names)
    assert not check_call("get_weather", {"unit": "celsius"}, TRUTH, names)
    assert not check_call("get_weather", {"city": "Calgary", "extra": 1}, TRUTH, names)


def test_check_call_maps_sent_names_back():
    truth = {"math.factorial": {"n": [5]}}
    assert check_call("math_factorial", {"n": 5.0}, truth, {"math_factorial": "math.factorial"})
    assert not check_call("math_factorial", {"n": True}, truth, {"math_factorial": "math.factorial"})


def test_score():
    irr = {"category": "irrelevance", "ground_truth": None}
    assert score(irr, [], {}) == 1.0
    assert score(irr, [("get_weather", {})], {}) == 0.0
    item = {"category": "simple", "ground_truth": [TRUTH]}
    assert score(item, [("get_weather", {"city": "calgary"})], {}) == 1.0
    assert score(item, [], {}) == 0.0


def test_convert_schema_maps_python_types():
    s = convert_schema({"type": "dict", "properties": {"n": {"type": "integer"}, "xs": {"type": "tuple"}}})
    assert s["type"] == "object"
    assert s["properties"]["xs"]["type"] == "array"


def test_validate():
    schema = {"properties": {"n": {"type": "integer"}, "mode": {"type": "string", "enum": ["a", "b"]}},
              "required": ["n"]}
    assert validate({"n": 3, "mode": "a"}, schema) == []
    assert len(validate({"n": True, "mode": "c", "x": 1}, schema)) == 3
    assert validate({}, schema) == ["missing required argument 'n'"]
