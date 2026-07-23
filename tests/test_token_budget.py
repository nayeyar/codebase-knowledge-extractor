from codebase_analyzer.token_budget import TokenBudgeter


def test_chunks_and_batches_stay_within_limits() -> None:
    budgeter = TokenBudgeter("gpt-5.4-mini")
    content = "\n".join(f"line {index}: value" for index in range(200))

    chunks = budgeter.chunk_file("Example.java", content, max_tokens=100)
    batches = budgeter.pack_batches(chunks, usable_tokens=180)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 100 for chunk in chunks)
    assert all(sum(chunk.token_count for chunk in batch) <= 180 for batch in batches)
    assert [chunk.part for chunk in chunks] == list(range(1, len(chunks) + 1))


def test_batching_accounts_for_rendered_metadata_overhead() -> None:
    budgeter = TokenBudgeter("gpt-5.4-mini")
    chunks = [
        budgeter.chunk_file("A.java", "value " * 40, max_tokens=100)[0],
        budgeter.chunk_file("B.java", "value " * 40, max_tokens=100)[0],
    ]

    batches = budgeter.pack_batches(
        chunks,
        usable_tokens=120,
        extra_tokens_by_chunk={chunk.chunk_id: 30 for chunk in chunks},
    )

    assert len(batches) == 2


def test_batching_limits_chunk_count_to_protect_output_budget() -> None:
    budgeter = TokenBudgeter("gpt-5.4-mini")
    chunks = [
        budgeter.chunk_file(f"{index}.java", "class A {}", max_tokens=100)[0] for index in range(5)
    ]

    batches = budgeter.pack_batches(chunks, usable_tokens=1_000, max_chunks_per_batch=2)

    assert [len(batch) for batch in batches] == [2, 2, 1]
