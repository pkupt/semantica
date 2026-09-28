"""Regression tests for #1788: _resolve_xsd must not double the xsd: prefix.

A range that already carries the prefix (``xsd:string``) or that is a full IRI
used to come back as ``xsd:xsd:string`` / ``xsd:http://...#string``, which is
not a resolvable IRI, so the generated ``sh:datatype`` constraint could never
match. Every range must resolve to a single, valid datatype IRI.
"""

import pytest

from semantica.ontology import SHACLGenerator


@pytest.fixture
def generator():
    return SHACLGenerator()


def test_already_xsd_qualified_range_is_left_alone(generator):
    assert generator._resolve_xsd("xsd:string") == "xsd:string"
    assert generator._resolve_xsd("xsd:integer") == "xsd:integer"


def test_bare_names_still_map_to_xsd(generator):
    assert generator._resolve_xsd("string") == "xsd:string"
    assert generator._resolve_xsd("int") == "xsd:integer"
    assert generator._resolve_xsd("datetime") == "xsd:dateTime"


def test_absolute_iri_range_is_left_alone(generator):
    iri = "http://www.w3.org/2001/XMLSchema#string"
    assert generator._resolve_xsd(iri) == iri


def test_scheme_without_slashes_is_left_alone(generator):
    # A URN or DOI is an absolute IRI too, even without "//".
    assert generator._resolve_xsd("urn:example:datatype") == "urn:example:datatype"
    assert generator._resolve_xsd("doi:10.1000/182") == "doi:10.1000/182"


def test_property_shape_carries_a_single_prefix(generator):
    shape = generator._build_property_shape(
        {"type": "datatype", "range": "xsd:string", "name": "title"}
    )
    assert shape.datatype == "xsd:string"

    bare = generator._build_property_shape(
        {"type": "datatype", "range": "string", "name": "title"}
    )
    assert bare.datatype == "xsd:string"
