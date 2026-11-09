import json

from maple_etf.graph import AgentGraph
from maple_etf.grounding import extract_numbers, ungrounded_numbers
from maple_etf.models import AccountType, Action, InvestorProfile


def test_extract_numbers():
    assert extract_numbers("MER 0.20%, fees $1,979, return -7.5%") == [0.20, 1979.0, -7.5]


def test_ungrounded_numbers_flags_invented_values():
    context = "MER is 0.20%. 12-month return is 24.8%."
    assert ungrounded_numbers("MER of 0.20% and a 24.8% return.", context) == []
    assert ungrounded_numbers("MER of 0.15%.", context) == [0.15]
    assert ungrounded_numbers("Three agents agree.", context) == []
    # Rounding a real number is fine; changing it is not.
    assert ungrounded_numbers("About 25% return.", "return 24.83") == []
    assert ungrounded_numbers("A 24.8% return.", "return 24.83") == []
    assert ungrounded_numbers("A 24.9% return.", "return 24.83") == [24.9]


def test_llm_quoting_real_facts_is_accepted(make_etf, fake_llm):
    llm = fake_llm("Fees are low: the MER is 0.20%, and holdings are spread across many regions.")
    result = AgentGraph(llm=llm).run(make_etf("XEQT.TO"), InvestorProfile())
    assert result.decision.explanation_source == "llm"


def test_tfsa_veto_blocks_us_listed_buy(make_etf):
    result = AgentGraph().run(make_etf("VOO", annual_return=0.25, annual_vol=0.08), InvestorProfile())
    assert result.decision.action != Action.BUY
    assert any("Tax/account veto" in v for v in result.decision.vetoes)


def test_same_etf_can_be_buy_in_rrsp(make_etf):
    etf = make_etf("VOO", annual_return=0.25, annual_vol=0.08)
    result = AgentGraph().run(etf, InvestorProfile(account=AccountType.RRSP, risk="high"))
    assert result.decision.action == Action.BUY
    assert 0 < result.decision.suggested_allocation_pct <= result.risk.max_allocation_pct


def test_llm_explanation_used_when_grounded(make_etf, fake_llm):
    llm = fake_llm("This is a low-cost, well diversified fund that fits a TFSA.")
    result = AgentGraph(llm=llm).run(make_etf("XEQT.TO"), InvestorProfile())
    assert result.decision.explanation_source == "llm"
    assert llm.usage["calls"] == 1


def test_llm_explanation_rejected_when_it_invents_numbers(make_etf, fake_llm):
    llm = fake_llm("The MER is only 0.01% and it returned 99.9% last year.")
    result = AgentGraph(llm=llm).run(make_etf("XEQT.TO"), InvestorProfile())
    assert result.decision.explanation_source == "llm_rejected"
    assert "0.01" not in result.decision.explanation


def test_llm_cannot_override_veto(make_etf, fake_llm):
    llm = fake_llm(json.dumps({"action": "BUY", "confidence": 99, "explanation": "Great fund, buy it."}))
    result = AgentGraph(llm=llm, llm_mode="decide").run(make_etf("VOO"), InvestorProfile())
    assert result.decision.action == Action.HOLD  # downgraded by the tax veto
    assert result.decision.decided_by == "llm"


def test_llm_failure_falls_back_to_rules(make_etf, fake_llm):
    llm = fake_llm(RuntimeError("network down"))
    result = AgentGraph(llm=llm, llm_mode="decide").run(make_etf("XEQT.TO"), InvestorProfile())
    assert result.decision.decided_by == "rules"
    assert result.decision.explanation_source == "template"


def test_bad_llm_json_falls_back(make_etf, fake_llm):
    llm = fake_llm("I think you should buy!")
    result = AgentGraph(llm=llm, llm_mode="decide").run(make_etf("XEQT.TO"), InvestorProfile())
    assert result.decision.decided_by == "rules"


def test_trace_file_written(tmp_path, make_etf):
    trace = tmp_path / "t.jsonl"
    AgentGraph(trace_path=trace).run(make_etf(), InvestorProfile())
    line = json.loads(trace.read_text().splitlines()[0])
    steps = [e["step"] for e in line["events"]]
    assert "portfolio_manager" in steps and "tax_account" in steps
