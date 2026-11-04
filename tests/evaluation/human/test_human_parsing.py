import json
from pathlib import Path

import numpy as np
import pytest

from hpe.evaluation.human import data, psych101


def test_parse_mcpl_words_and_feedback_phases():
    from hv_fixtures import mcpl_text

    txt = mcpl_text(np.random.default_rng(1), n_trials=12, n_feedback=8)
    p = psych101.parse_mcpl(txt, "collsiöö2023MCPL/exp1.csv", "3")
    assert p.paradigm == "mcpl" and p.n_actions == 9 and len(p.trials) == 12
    assert sum(t.correct is not None for t in p.trials) == 8
    assert all(0 <= t.cue1 < 5 and 0 <= t.choice < 9 for t in p.trials)
    rep = psych101.validate_mcpl([p])
    assert rep["collsiöö2023MCPL/exp1.csv"]["no_feedback_trials"] == 4


def test_parse_mcpl_numeric_levels():
    from hv_fixtures import mcpl_text

    p = psych101.parse_mcpl(
        mcpl_text(np.random.default_rng(2), numeric=True), "collsiöö2023MCPL/exp2.csv", "0"
    )
    assert p.meta["cue_levels"] == ["1", "2", "3", "4", "5"] and len(p.trials) == 40


def test_parse_mcpl_rejects_malformed_and_inconsistent():
    from hv_fixtures import mcpl_text

    txt = mcpl_text(np.random.default_rng(3), n_trials=3, n_feedback=3)
    with pytest.raises(psych101.ParseError):
        psych101.parse_mcpl(txt + "\nProgladine: average. <<garbage", "e", "0")
    bad = (
        txt.replace("That is incorrect", "That is correct")
        if "incorrect" in txt
        else txt.replace("That is correct", "That is incorrect")
    )
    p = psych101.parse_mcpl(bad, "e", "0")
    with pytest.raises(psych101.ParseError):
        psych101.validate_mcpl([p])


def test_parse_igt_maps_labels_by_instruction_order():
    txt = (
        "You see in front of you four decks of cards labeled Q, W, E, and R.\n\n"
        "You press <<E>>. You win 50.0$ and lose 0.0$.\nYou press <<Q>>. You win 100.0$ and lose 250.0$."
    )
    p = psych101.parse_igt(txt, "steingroever2015data/exp1.csv", "0")
    assert [t.choice for t in p.trials] == [2, 0]
    assert p.trials[1].net == -150.0
    with pytest.raises(psych101.ParseError):
        psych101.parse_igt(txt + "\nYou press <<Z>>. You win 1.0$ and lose 0.0$.", "e", "0")


def test_load_participants_filters_and_sorts(psych_jsonl: Path):
    m = psych101.load_participants(psych_jsonl, "mcpl")
    g = psych101.load_participants(psych_jsonl, "igt")
    assert len(m) == 18 and len(g) == 14
    assert [p.participant for p in g] == [str(i) for i in range(14)]
    rep = psych101.validate_igt(g)
    assert rep["experiments"]["steingroever2015data/exp1.csv"]["trials_per_participant"] == [30]
    assert rep["median_win_by_deck_position"][0] == 100.0


def test_filter_jsonl_preserves_bytes_and_order(psych_jsonl: Path, tmp_path: Path):
    out = tmp_path / "sub.jsonl"
    n = data.filter_jsonl(psych_jsonl, out, ["steingroever2015data/"])
    assert n == 14
    src_lines = [
        ln
        for ln in psych_jsonl.read_text(encoding="utf-8").splitlines(True)
        if "steingroever" in json.loads(ln)["experiment"]
    ]
    assert out.read_text(encoding="utf-8").splitlines(True) == src_lines


def test_hash_verification_and_locate(tmp_path: Path, monkeypatch):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    digest = data.sha256_file(f)
    assert data.verify(f, digest) == digest
    with pytest.raises(data.HashMismatch):
        data.verify(f, "0" * 64)
    src = data.DatasetSource(
        "x", "file://none", digest, None, "CC0", "test", "test", "2026-11-04", "HPE_TEST_X"
    )
    monkeypatch.setenv("HPE_TEST_X", str(f))
    assert data.locate(src) == f
    monkeypatch.delenv("HPE_TEST_X")
    monkeypatch.setattr(data, "EXTERNAL_DIR", tmp_path / "nope")
    with pytest.raises(data.DatasetUnavailable):
        data.locate(src)


def test_download_from_file_url_verifies(tmp_path: Path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello human data")
    dest = tmp_path / "out" / "dest.bin"
    d = data.download(src.as_uri(), dest, data.sha256_file(src))
    assert dest.read_bytes() == b"hello human data" and d == data.sha256_file(dest)
    with pytest.raises(data.HashMismatch):
        data.download(src.as_uri(), tmp_path / "bad.bin", "f" * 64)
    assert not (tmp_path / "bad.bin").exists()


def test_recorded_sources_have_licences():
    for s in (data.PSYCH101_FULL, data.PSYCH101_LEARNING_SUBSET, data.HEARTSTEPS_SUGGESTIONS):
        assert s.licence and s.citation and s.sha256 and len(s.sha256) == 64
