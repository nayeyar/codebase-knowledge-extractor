from __future__ import annotations

import re
from collections.abc import Iterator

import tree_sitter_java
from tree_sitter import Language, Node, Parser

from codebase_analyzer.models import JavaFileAnalysis, MethodFact

JAVA_LANGUAGE = Language(tree_sitter_java.language())
JAVA_PARSER = Parser(JAVA_LANGUAGE)

CLASS_NODE_TYPES = {
    "class_declaration",
    "enum_declaration",
    "interface_declaration",
    "record_declaration",
}

DECISION_NODE_TYPES = {
    "if_statement",
    "for_statement",
    "enhanced_for_statement",
    "while_statement",
    "do_statement",
    "catch_clause",
    "conditional_expression",
    "switch_label",
}

VISIBILITY_MODIFIERS = ("public", "protected", "private")
SIGNATURE_MODIFIERS = {
    "public",
    "protected",
    "private",
    "static",
    "final",
    "abstract",
    "synchronized",
    "native",
    "default",
    "strictfp",
}


def parse_java_file(file_path: str, content: str) -> JavaFileAnalysis:
    source = content.encode("utf-8")
    tree = JAVA_PARSER.parse(source)
    package_match = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", content)
    package = package_match.group(1) if package_match else ""

    classes: list[str] = []
    method_nodes: list[Node] = []
    for node in walk_named(tree.root_node):
        if node.type in CLASS_NODE_TYPES:
            name = node.child_by_field_name("name")
            if name:
                classes.append(_text(name, source))
        elif node.type in {"method_declaration", "constructor_declaration"}:
            method_nodes.append(node)

    methods = [
        _method_fact(index, file_path, method_node, source)
        for index, method_node in enumerate(method_nodes, start=1)
    ]
    return JavaFileAnalysis(
        file_path=file_path,
        package=package,
        classes=list(dict.fromkeys(classes)),
        line_count=content.count("\n") + (1 if content else 0),
        methods=methods,
    )


def walk_named(root: Node) -> Iterator[Node]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.named_children))


def complexity_rating(value: int) -> str:
    if value <= 5:
        return "low"
    if value <= 10:
        return "moderate"
    if value <= 20:
        return "high"
    return "very_high"


def _method_fact(index: int, file_path: str, node: Node, source: bytes) -> MethodFact:
    name_node = node.child_by_field_name("name")
    name = _text(name_node, source) if name_node else "<anonymous>"
    class_name = _nearest_class_name(node, source)
    modifiers_node = next((child for child in node.children if child.type == "modifiers"), None)
    modifier_text = _text(modifiers_node, source) if modifiers_node else ""
    modifier_words = [
        word for word in re.findall(r"\b\w+\b", modifier_text) if word in SIGNATURE_MODIFIERS
    ]
    visibility = next((item for item in VISIBILITY_MODIFIERS if item in modifier_words), "package")
    annotations = _annotation_names(modifiers_node, source)
    signature = _build_signature(node, source, modifier_words)
    complexity = _cyclomatic_complexity(node, source)
    stable_path = file_path.removesuffix(".java").replace("/", ".")
    method_id = f"{stable_path}#{class_name}.{name}:{index}"

    return MethodFact(
        method_id=method_id,
        file_path=file_path,
        class_name=class_name,
        name=name,
        signature=signature,
        start_line=node.start_point.row + 1,
        end_line=node.end_point.row + 1,
        cyclomatic_complexity=complexity,
        complexity_rating=complexity_rating(complexity),
        annotations=annotations,
        visibility=visibility,
    )


def _nearest_class_name(node: Node, source: bytes) -> str:
    current = node.parent
    while current:
        if current.type in CLASS_NODE_TYPES:
            name = current.child_by_field_name("name")
            return _text(name, source) if name else "<anonymous>"
        current = current.parent
    return "<top-level>"


def _annotation_names(modifiers_node: Node | None, source: bytes) -> list[str]:
    if modifiers_node is None:
        return []
    annotations: list[str] = []
    for node in walk_named(modifiers_node):
        if node.type in {"annotation", "marker_annotation"}:
            name = node.child_by_field_name("name")
            if name:
                annotations.append(_text(name, source))
    return annotations


def _build_signature(node: Node, source: bytes, modifier_words: list[str]) -> str:
    body = node.child_by_field_name("body")
    end_byte = body.start_byte if body else node.end_byte
    declaration = source[node.start_byte : end_byte].decode("utf-8", errors="replace")
    declaration = re.sub(r"(?s)^\s*(?:@[\w.]+(?:\([^)]*\))?\s*)+", "", declaration)
    declaration = re.sub(r"\s+", " ", declaration).strip().rstrip(";").strip()
    if modifier_words and not any(declaration.startswith(word + " ") for word in modifier_words):
        declaration = " ".join(modifier_words) + " " + declaration
    return declaration


def _cyclomatic_complexity(method_node: Node, source: bytes) -> int:
    complexity = 1
    body = method_node.child_by_field_name("body")
    if body is None:
        return complexity

    for node in walk_named(body):
        if node.type in DECISION_NODE_TYPES:
            if node.type == "switch_label" and _text(node, source).lstrip().startswith("default"):
                continue
            complexity += 1
        elif node.type == "binary_expression":
            operators = [
                _text(child, source)
                for child in node.children
                if not child.is_named and child.type in {"&&", "||"}
            ]
            complexity += len(operators)
    return complexity


def _text(node: Node | None, source: bytes) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
