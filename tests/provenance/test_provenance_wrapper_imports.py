"""
Regression tests for provenance wrapper import resolution.

Every ``*WithProvenance`` wrapper imports its underlying implementation lazily,
inside ``__init__``.  A wrong import path is therefore invisible at import time
and only surfaces when the wrapper is instantiated.  The surrounding suites
guard those instantiations with ``except ImportError: pytest.skip(...)``, which
converts the failure into a skip and hides it from CI.

These tests intentionally do not skip: if a wrapper cannot resolve its
underlying implementation, the test fails.
"""


class TestIngestProvenanceWrapper:
    """PDFIngestorWithProvenance must resolve the file ingestor it wraps."""

    def test_instantiates_and_wraps_file_ingestor(self):
        from semantica.ingest.file_ingestor import FileIngestor
        from semantica.ingest.ingest_provenance import PDFIngestorWithProvenance

        wrapper = PDFIngestorWithProvenance(provenance=False)
        assert isinstance(wrapper._ingestor, FileIngestor)

    def test_delegates_unknown_attributes(self):
        from semantica.ingest.ingest_provenance import PDFIngestorWithProvenance

        wrapper = PDFIngestorWithProvenance(provenance=False)
        # `ingest_file` is defined on FileIngestor, not on the wrapper.
        assert callable(wrapper.ingest_file)


class TestParseProvenanceWrapper:
    """ParserWithProvenance must resolve the document parser it wraps."""

    def test_instantiates_and_wraps_document_parser(self):
        from semantica.parse.document_parser import DocumentParser
        from semantica.parse.parse_provenance import ParserWithProvenance

        wrapper = ParserWithProvenance(provenance=False)
        assert isinstance(wrapper._parser, DocumentParser)

    def test_delegates_unknown_attributes(self):
        from semantica.parse.parse_provenance import ParserWithProvenance

        wrapper = ParserWithProvenance(provenance=False)
        # `parse_document` is defined on DocumentParser, not on the wrapper.
        assert callable(wrapper.parse_document)
