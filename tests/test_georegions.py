"""Tests for georegion query expansion and result annotation.

These tests verify that the georegion functionality ported from the original
vastai_sdk (commit 6575803) works identically to the old implementation.
"""

import pytest
from vastai.utils import (
    _regions,
    _regions_rev,
    expand_georegion_query,
    annotate_georegion_results,
)


class TestRegionsMappings:
    """Verify the region data itself is correct."""

    def test_all_region_codes_present(self):
        assert set(_regions.keys()) == {'AF', 'AS', 'EU', 'LC', 'NA', 'OC'}

    def test_na_countries(self):
        assert _regions['NA'] == 'CA,US'

    def test_reverse_mapping_us(self):
        assert _regions_rev['US'] == 'NA'

    def test_reverse_mapping_de(self):
        assert _regions_rev['DE'] == 'EU'

    def test_reverse_mapping_jp(self):
        assert _regions_rev['JP'] == 'AS'

    def test_reverse_mapping_br(self):
        assert _regions_rev['BR'] == 'LC'

    def test_reverse_mapping_au(self):
        # AU appears in both AS and OC in the original data; OC is processed
        # last so it wins in the reverse mapping.
        assert _regions_rev['AU'] in ('AS', 'OC')

    def test_reverse_mapping_za(self):
        assert _regions_rev['ZA'] == 'AF'

    def test_every_country_maps_back(self):
        """Every country in _regions must appear in _regions_rev."""
        for code, countries_str in _regions.items():
            for country in countries_str.split(','):
                assert country in _regions_rev, f"{country} from region {code} not in reverse mapping"


class TestExpandGeoregionQuery:
    """Test the pre-processing query expansion hook."""

    def test_none_query(self):
        active, q = expand_georegion_query(None)
        assert active is False
        assert q is None

    def test_no_georegion_flag(self):
        active, q = expand_georegion_query('num_gpus = 1 gpu_name = RTX_4090')
        assert active is False
        assert q == 'num_gpus = 1 gpu_name = RTX_4090'

    def test_georegion_na_expansion(self):
        active, q = expand_georegion_query('num_gpus = 1 geolocation = NA georegion = true')
        assert active is True
        assert 'georegion' not in q
        assert 'geolocation in [CA,US]' in q
        assert 'num_gpus = 1' in q

    def test_georegion_eu_expansion(self):
        active, q = expand_georegion_query('geolocation = EU georegion = true')
        assert active is True
        assert 'geolocation in [' in q
        assert 'DE' in q
        assert 'FR' in q
        assert 'GB' in q

    def test_georegion_preserves_other_fields(self):
        active, q = expand_georegion_query('num_gpus = 2 gpu_name = A100 geolocation = NA georegion = true')
        assert active is True
        assert 'num_gpus = 2' in q
        assert 'gpu_name = A100' in q

    def test_georegion_false_not_set(self):
        """georegion=false or georegion=anything-else should not trigger expansion."""
        active, q = expand_georegion_query('geolocation = NA georegion = false')
        assert active is False

    def test_georegion_without_geolocation(self):
        """georegion=true but no geolocation field — just strips georegion."""
        active, q = expand_georegion_query('num_gpus = 1 georegion = true')
        assert active is True
        assert 'georegion' not in q
        assert 'num_gpus = 1' in q


class TestAnnotateGeoregionResults:
    """Test the post-processing result annotation hook."""

    def test_annotate_us(self):
        results = [{'geolocation': 'US', 'id': 1}]
        annotated = annotate_georegion_results(results)
        assert annotated[0]['geolocation'] == 'US, NA'

    def test_annotate_de(self):
        results = [{'geolocation': 'DE', 'id': 2}]
        annotated = annotate_georegion_results(results)
        assert annotated[0]['geolocation'] == 'DE, EU'

    def test_annotate_multiple(self):
        results = [
            {'geolocation': 'US', 'id': 1},
            {'geolocation': 'DE', 'id': 2},
            {'geolocation': 'JP', 'id': 3},
        ]
        annotated = annotate_georegion_results(results)
        assert annotated[0]['geolocation'] == 'US, NA'
        assert annotated[1]['geolocation'] == 'DE, EU'
        assert annotated[2]['geolocation'] == 'JP, AS'

    def test_annotate_unknown_country(self):
        """Unknown countries should be left unchanged (not crash)."""
        results = [{'geolocation': 'XX', 'id': 1}]
        annotated = annotate_georegion_results(results)
        assert annotated[0]['geolocation'] == 'XX'

    def test_annotate_empty_geolocation(self):
        results = [{'geolocation': '', 'id': 1}]
        annotated = annotate_georegion_results(results)
        assert annotated[0]['geolocation'] == ''

    def test_annotate_missing_geolocation(self):
        results = [{'id': 1}]
        annotated = annotate_georegion_results(results)
        assert 'geolocation' not in annotated[0] or annotated[0].get('geolocation', '') == ''

    def test_annotate_modifies_in_place(self):
        """Results list is modified in place (matching old behavior)."""
        results = [{'geolocation': 'CA', 'id': 1}]
        returned = annotate_georegion_results(results)
        assert returned is results
        assert results[0]['geolocation'] == 'CA, NA'

    def test_annotate_longer_geolocation_string(self):
        """Handles geolocation values like 'California, US' by taking last 2 chars."""
        results = [{'geolocation': 'California, US', 'id': 1}]
        annotated = annotate_georegion_results(results)
        assert annotated[0]['geolocation'] == 'California, US, NA'


class TestEndToEndGeoregionFlow:
    """Test the full expand → parse_query → annotate pipeline."""

    def test_full_pipeline_na(self):
        from vastai.api.query import parse_query, offers_fields, offers_alias, offers_mult

        # Step 1: Expand
        active, query_str = expand_georegion_query(
            'num_gpus = 1 geolocation = NA georegion = true'
        )
        assert active is True

        # Step 2: Parse into query dict
        query = parse_query(query_str, {}, offers_fields, offers_alias, offers_mult)
        assert 'geolocation' in query
        assert 'in' in query['geolocation']
        countries = query['geolocation']['in']
        assert 'CA' in countries
        assert 'US' in countries

        # Step 3: Annotate results
        fake_results = [
            {'geolocation': 'US', 'id': 1},
            {'geolocation': 'CA', 'id': 2},
        ]
        annotated = annotate_georegion_results(fake_results)
        assert annotated[0]['geolocation'] == 'US, NA'
        assert annotated[1]['geolocation'] == 'CA, NA'

    def test_full_pipeline_no_georegion(self):
        from vastai.api.query import parse_query, offers_fields, offers_alias, offers_mult

        active, query_str = expand_georegion_query('num_gpus = 1 gpu_name = RTX_4090')
        assert active is False

        query = parse_query(query_str, {}, offers_fields, offers_alias, offers_mult)
        assert 'georegion' not in query
        assert query['num_gpus'] == {'eq': '1'}
