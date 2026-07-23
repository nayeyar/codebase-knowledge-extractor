from pathlib import Path

from codebase_analyzer.java_parser import parse_java_file

FIXTURE = Path(__file__).parent / "fixtures" / "SampleController.java"


def test_extracts_method_signature_annotations_and_complexity() -> None:
    analysis = parse_java_file("SampleController.java", FIXTURE.read_text(encoding="utf-8"))

    assert analysis.package == "com.example.sample"
    assert analysis.classes == ["SampleController"]
    assert len(analysis.methods) == 1
    method = analysis.methods[0]
    assert method.name == "findActive"
    assert method.class_name == "SampleController"
    assert method.signature == "public List<String> findActive(List<String> values)"
    assert method.annotations == ["GetMapping"]
    assert method.visibility == "public"
    assert method.cyclomatic_complexity == 4
    assert method.complexity_rating == "low"
