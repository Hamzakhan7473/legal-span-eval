from lse.data.eval_set import (
    EVAL_SET_PATH,
    load_config,
    read_cuad_ids,
    read_eval_set,
    sha256_file,
    verify_manifest,
)


def test_eval_set_is_byte_identical_to_the_evaluated_file():
    assert sha256_file(EVAL_SET_PATH) == load_config()["expected_sha256"]


def test_manifest_matches_committed_files():
    assert verify_manifest() == []


def test_composition_matches_construction_parameters():
    construction = load_config()["construction"]
    examples = read_eval_set()
    categories = sorted({e["category"] for e in examples})
    assert len(categories) == construction["n_categories"]
    assert len(examples) == construction["n_categories"] * construction["n_per_category"]
    for category in categories:
        rows = [e for e in examples if e["category"] == category]
        assert sum(e["present"] for e in rows) == construction["n_present_per_category"]
        assert sum(not e["present"] for e in rows) == construction["n_absent_per_category"]


def test_ids_are_unique_and_aligned():
    examples = read_eval_set()
    cuad_ids = read_cuad_ids()
    assert len({e["id"] for e in examples}) == len(examples)
    assert len(set(cuad_ids)) == len(cuad_ids) == len(examples)
    for example, cuad_id in zip(examples, cuad_ids, strict=True):
        assert cuad_id.endswith(f"__{example['category']}")


def test_present_examples_have_gold_spans():
    for example in read_eval_set():
        assert bool(example["gold_spans"]) == example["present"]
