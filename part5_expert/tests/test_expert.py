# part5_expert/tests/test_expert.py
# Unit tests for Module E — TC-E1 through TC-E4.

import pytest
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from part5_expert.forward_chaining import diagnose
from part5_expert.add_knowledge import append_rule
from part5_expert.explanation import format_trace


_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rules.json"
)


# ---------------------------------------------------------------------------
# TC-E1 — Single-rule fire
# ---------------------------------------------------------------------------
class TestTCE1:
    def test_battery_fault(self):
        facts = {"symptom:battery_voltage_low", "symptom:battery_drain_fast"}
        result = diagnose(facts)
        assert "diagnosis:battery_fault" in result["diagnoses"], (
            f"Expected diagnosis:battery_fault in {result['diagnoses']}"
        )
        assert "R3" in result["fired_rule_ids"], (
            f"Expected R3 in fired_rule_ids: {result['fired_rule_ids']}"
        )


# ---------------------------------------------------------------------------
# TC-E2 — Multi-step chain (R4 → R5)
# ---------------------------------------------------------------------------
class TestTCE2:
    def test_chain_r4_r5(self):
        facts = {"symptom:battery_temp_high", "symptom:battery_voltage_low"}
        result = diagnose(facts)

        # Both must appear in working memory
        assert "derived:power_subsystem_stressed" in result["working_memory"], (
            f"derived:power_subsystem_stressed missing. WM: {result['working_memory']}"
        )
        assert "diagnosis:power_subsystem_failure" in result["working_memory"], (
            f"diagnosis:power_subsystem_failure missing. WM: {result['working_memory']}"
        )

        # R4 must fire before R5 in the trace
        trace_ids = [t[0] for t in result["trace"]]
        assert "R4" in trace_ids and "R5" in trace_ids, f"trace_ids: {trace_ids}"
        idx_r4 = trace_ids.index("R4")
        idx_r5 = trace_ids.index("R5")
        assert idx_r4 < idx_r5, (
            f"R4 must appear before R5 in trace. Got order: {trace_ids}"
        )


# ---------------------------------------------------------------------------
# TC-E3 — No match
# ---------------------------------------------------------------------------
class TestTCE3:
    def test_no_facts(self):
        result = diagnose(set())
        assert len(result["fired_rule_ids"]) == 0, (
            f"Expected no rules fired, got: {result['fired_rule_ids']}"
        )
        assert len(result["diagnoses"]) == 0, (
            f"Expected no diagnoses, got: {result['diagnoses']}"
        )


# ---------------------------------------------------------------------------
# TC-E4 — Knowledge acquisition round-trip
# ---------------------------------------------------------------------------
class TestTCE4:
    def test_ka_round_trip(self, tmp_path):
        """
        Programmatically add R19 to a copy of rules.json, reload the engine,
        assert diagnosis:custom_test_fault is reached — no .py file change needed.
        """
        # Copy original rules.json to a temp file
        temp_rules = str(tmp_path / "rules.json")
        shutil.copy(_RULES_PATH, temp_rules)

        new_rule = {
            "id": "R19",
            "if": ["symptom:custom_test_flag"],
            "not": [],
            "then": "diagnosis:custom_test_fault",
            "explanation": "test",
        }

        # Use the importable append_rule function (no .py change required)
        append_rule(new_rule, temp_rules)

        # Verify rule was written
        with open(temp_rules) as f:
            rules = json.load(f)
        assert any(
            r["id"] == "R19" for r in rules), "R19 not found in temp rules.json"

        # Run the engine with the temp rules file
        result = diagnose({"symptom:custom_test_flag"}, rules_path=temp_rules)
        assert "diagnosis:custom_test_fault" in result["diagnoses"], (
            f"Expected diagnosis:custom_test_fault, got: {result['diagnoses']}"
        )


# ---------------------------------------------------------------------------
# Explanation facility
# ---------------------------------------------------------------------------
class TestExplanation:
    def test_format_trace_order(self):
        trace = [
            ("R4", "High battery temperature is an early sign of power subsystem stress.",
             "derived:power_subsystem_stressed"),
            ("R5", "Low voltage combined with confirmed power stress.",
             "diagnosis:power_subsystem_failure"),
        ]
        output = format_trace(trace)
        lines = output.strip().split("\n")
        assert "R4" in lines[0]
        assert "R5" in lines[1]
        assert "derived:power_subsystem_stressed" in lines[0]


# ---------------------------------------------------------------------------
# Rule file integrity
# ---------------------------------------------------------------------------
class TestRuleFile:
    def test_exactly_18_rules(self):
        with open(_RULES_PATH) as f:
            rules = json.load(f)
        assert len(rules) == 18, f"Expected 18 rules, found {len(rules)}"

    def test_rule_ids_r1_to_r18(self):
        with open(_RULES_PATH) as f:
            rules = json.load(f)
        ids = {r["id"] for r in rules}
        expected = {f"R{i}" for i in range(1, 19)}
        assert ids == expected, f"Rule IDs mismatch: {ids}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
