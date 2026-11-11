import json

from maple_etf.judge import HeuristicJudge, Item, LLMJudge, calibrate, cohen_kappa, label_interactively, load_items

FACTS = "MER is 0.20%, which is low. Holds about 9,000 securities. Decision: BUY, weighted score +0.62."


def item(text, decision="BUY", human=None):
    return Item("t", "XEQT.TO", decision, FACTS, text, human or {"correct": True, "clear": True, "simple": True})


def test_heuristic_catches_wrong_number_and_wrong_decision():
    j = HeuristicJudge()
    assert j.grade(item("BUY XEQT. The MER is only 0.20% and it holds about 9,000 companies.")).correct
    assert not j.grade(item("BUY XEQT. The MER is only 0.15% and it holds about 9,000 companies.")).correct
    assert not j.grade(item("AVOID XEQT. The MER is 0.20% and it holds about 9,000 companies.")).correct


def test_heuristic_flags_jargon_and_long_sentences():
    j = HeuristicJudge()
    assert not j.grade(item("BUY XEQT because its Sharpe ratio and drawdown look good for most people.")).simple
    long = "BUY XEQT " + " ".join(["because it is good"] * 10) + "."
    assert not j.grade(item(long)).simple


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0


def test_llm_judge_parses_json(fake_llm):
    llm = fake_llm(json.dumps({"correct": False, "clear": True, "simple": True, "why": "wrong fee"}))
    g = LLMJudge(llm).grade(item("BUY XEQT, fee 0.15%."))
    assert (g.correct, g.clear, g.simple) == (False, True, True)


def test_calibration_on_starter_labels():
    res = calibrate(HeuristicJudge(), load_items())
    assert res["items"] == 15
    assert res["criteria"]["correct"]["agreement"] >= 0.8
    # The heuristic can't see a wrong claim without numbers - that's why we want an LLM judge.
    assert any(d["id"] == "vdy-wrong-claim" for d in res["disagreements"])


def test_label_interactively(tmp_path, monkeypatch):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"items": [{"id": "a", "ticker": "XEQT.TO", "decision": "BUY", "facts": FACTS,
                                           "explanation": "BUY XEQT.", "human": {}, "label_source": ""}]}))
    answers = iter(["y", "n", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert label_interactively(load_items(path), path, who="shivani") == 1
    saved = json.loads(path.read_text())["items"][0]
    assert saved["human"] == {"correct": True, "clear": False, "simple": True}
    assert saved["label_source"] == "shivani"
