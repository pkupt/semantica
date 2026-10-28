"""
Tests for the BridgeAxiom applicability window, units and uncertainty.

Issue #2138: a coefficient was recorded without the unit it is expressed in,
the range the source covers, or any spread measure, so a value carried across
layers could be extrapolated without anyone noticing.
"""

import pytest

from semantica.provenance import ProvenanceManager
from semantica.provenance.bridge_axiom import BridgeAxiom


def make_axiom(**overrides):
    kwargs = dict(
        axiom_id="BA-TEST-2138",
        name="biomass_tourism_elasticity",
        rule="1% biomass increase -> 0.346% tourism revenue increase",
        coefficient=0.346,
        source_doi="10.1038/s41586-021-03371-z",
        source_page="p4",
        source_quote="a 1% increase in biomass raises revenue by 0.346%",
    )
    kwargs.update(overrides)
    return BridgeAxiom(**kwargs)


class TestOptionalFields:
    def test_fields_default_to_none(self):
        ba = make_axiom()
        assert ba.units is None
        assert ba.applicability_min is None
        assert ba.applicability_max is None
        assert ba.uncertainty is None

    def test_fields_are_kept(self):
        ba = make_axiom(
            units="percent_change",
            applicability_min=0.0,
            applicability_max=10.0,
            uncertainty=0.041,
        )
        assert ba.units == "percent_change"
        assert ba.applicability_min == 0.0
        assert ba.applicability_max == 10.0
        assert ba.uncertainty == 0.041

    def test_to_dict_carries_the_fields(self):
        ba = make_axiom(
            units="ratio",
            applicability_min=1.0,
            applicability_max=5.0,
            uncertainty=0.01,
        )
        as_dict = ba.to_dict()
        assert as_dict["units"] == "ratio"
        assert as_dict["applicability_min"] == 1.0
        assert as_dict["applicability_max"] == 5.0
        assert as_dict["uncertainty"] == 0.01

    def test_positional_construction_is_unchanged(self):
        """The new fields are appended after metadata, so the documented
        positional order still builds the same axiom."""
        ba = BridgeAxiom(
            "BA-POS-1",
            "positional",
            "rule",
            0.5,
            "10.1234/x",
            "p1",
            "quote",
        )
        assert ba.source_quote == "quote"
        assert ba.units is None


class TestWindowValidation:
    def test_inverted_window_is_rejected(self):
        with pytest.raises(ValueError, match="applicability_min"):
            make_axiom(applicability_min=10.0, applicability_max=0.0)

    def test_equal_bounds_are_allowed(self):
        ba = make_axiom(applicability_min=2.0, applicability_max=2.0)
        assert ba.applicability_violations(2.0) == []

    def test_negative_uncertainty_is_rejected(self):
        with pytest.raises(ValueError, match="uncertainty"):
            make_axiom(uncertainty=-0.1)

    def test_open_ended_window_is_allowed(self):
        below = make_axiom(applicability_max=5.0)
        assert below.applicability_violations(100.0) != []
        above = make_axiom(applicability_min=1.0)
        assert above.applicability_violations(0.0) != []


class TestExtrapolationReporting:
    def test_inside_window_is_not_extrapolated(self):
        ba = make_axiom(applicability_min=0.0, applicability_max=10.0)
        assert ba.applicability_violations(4.0) == []
        result = ba.apply("biomass", 4.0)
        assert result["extrapolated"] is False
        assert result["applicability_violations"] == []

    def test_below_minimum_is_reported(self):
        ba = make_axiom(applicability_min=0.0, applicability_max=10.0)
        result = ba.apply("biomass", -1.0)
        assert result["extrapolated"] is True
        assert len(result["applicability_violations"]) == 1
        assert "below applicability_min" in result["applicability_violations"][0]

    def test_above_maximum_is_reported(self):
        ba = make_axiom(applicability_min=0.0, applicability_max=10.0)
        result = ba.apply("biomass", 12.5)
        assert result["extrapolated"] is True
        assert len(result["applicability_violations"]) == 1
        assert "above applicability_max" in result["applicability_violations"][0]

    def test_bounds_are_inclusive(self):
        ba = make_axiom(applicability_min=0.0, applicability_max=10.0)
        assert ba.apply("biomass", 0.0)["extrapolated"] is False
        assert ba.apply("biomass", 10.0)["extrapolated"] is False

    def test_no_window_never_reports_extrapolation(self):
        ba = make_axiom()
        result = ba.apply("biomass", 1e9)
        assert result["extrapolated"] is False
        assert result["applicability_violations"] == []

    def test_the_output_value_is_still_returned(self):
        """Reporting extrapolation must not silently change the number: the
        caller decides what to do with it."""
        ba = make_axiom(coefficient=2.0, applicability_min=0.0, applicability_max=10.0)
        result = ba.apply("x", 50.0)
        assert result["output_value"] == 100.0


class TestResultAndProvenance:
    def test_result_carries_units_and_uncertainty(self):
        ba = make_axiom(units="percent_change", uncertainty=0.041)
        result = ba.apply("biomass", 3.0)
        assert result["units"] == "percent_change"
        assert result["uncertainty"] == 0.041

    def test_provenance_records_the_window_and_the_flag(self):
        prov_mgr = ProvenanceManager()
        ba = make_axiom(
            units="percent_change",
            applicability_min=0.0,
            applicability_max=10.0,
            uncertainty=0.041,
        )
        result = ba.apply("biomass", 12.5, prov_manager=prov_mgr)

        lineage = prov_mgr.get_lineage(result["output_entity"])
        assert lineage is not None

        recorded = lineage.get("metadata") or {}
        assert recorded.get("axiom_id") == "BA-TEST-2138"
        assert recorded["units"] == "percent_change"
        assert recorded["applicability_max"] == 10.0
        assert recorded["extrapolated"] is True
        assert recorded["applicability_violations"]
