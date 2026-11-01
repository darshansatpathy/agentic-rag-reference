from agentic_rag.text import (
    answer_type,
    has_quantity,
    has_temporal,
    normalise,
    numbers,
    sentences,
    stem,
    tokens,
)


def test_normalise_drops_thousands_separators() -> None:
    assert normalise("Up to 3,000 EUR") == "up to 3000 eur"
    assert normalise("1,5") == "1,5"  # not a thousands separator


def test_stemming_matches_inflections() -> None:
    assert stem("approved") == stem("approve") == stem("approval")
    assert stem("days") == stem("day")
    assert stem("acknowledged") == stem("acknowledge")
    assert stem("2026") == "2026"
    assert stem("api") == "api"  # short tokens untouched


def test_tokens_drop_stopwords_and_single_letters() -> None:
    assert tokens("What is the per diem in Belgium?") == ["diem", "belgium"]
    assert "a" not in tokens("a b c 5")
    assert "5" in tokens("a b c 5")
    assert "the" in tokens("the end", keep_stopwords=True)


def test_numbers_and_sentences() -> None:
    assert numbers("Between 500 and 5,000 EUR, SEV2") == {"500", "5000", "sev2"}
    assert sentences("One. Two! 3 is next? yes") == ["One.", "Two!", "3 is next? yes"]
    assert sentences("   ") == []


def test_answer_type_detection() -> None:
    assert answer_type("When will the site close?") == "time"
    assert answer_type("How many dock doors are there?") == "quantity"
    assert answer_type("What is the rate limit?") == "quantity"
    assert answer_type("Who approves expenses?") is None


def test_temporal_and_quantity_cues() -> None:
    assert has_temporal("The review is due within five working days.")
    assert has_temporal("It closes on 30 September 2027.")
    assert not has_temporal("The cross-dock has 12,000 square metres.")
    assert has_quantity("six months")
    assert not has_quantity("the cost-centre owner")
