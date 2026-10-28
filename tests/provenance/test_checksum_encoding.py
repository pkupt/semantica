"""Checksum encoding tests (issue #2136).

`compute_checksum()` used to paste the 16 hashed fields together with nothing
between them, so two records differing only in where a field boundary falls
produced the same hash. These tests pin the injective encoding, the per-row
version dispatch that keeps pre-existing stores verifiable, and the end-to-end
detection path through `verify_chain()`.
"""

import sqlite3

from semantica.provenance.integrity import (
    CHECKSUM_VERSION,
    LEGACY_CHECKSUM_VERSION,
    checksum_version,
    compute_checksum,
    compute_checksum_legacy,
    verify_checksum,
)
from semantica.provenance.manager import ProvenanceManager
from semantica.provenance.schemas import ProvenanceEntry


class TestInjectiveEncoding:
    def test_field_boundary_move_changes_checksum(self):
        """`ab` + `c` must not hash like `a` + `bc`."""
        a = {"entity_type": "ab", "activity_id": "c"}
        b = {"entity_type": "a", "activity_id": "bc"}
        assert compute_checksum(a) != compute_checksum(b)

    def test_field_boundary_move_collides_under_legacy(self):
        """Documents the v1 hole the fix closes, so a regression to plain
        concatenation fails here rather than silently passing."""
        a = {"entity_type": "ab", "activity_id": "c"}
        b = {"entity_type": "a", "activity_id": "bc"}
        assert compute_checksum_legacy(a) == compute_checksum_legacy(b)

    def test_list_element_boundary_move_changes_checksum(self):
        """One element containing a comma must not hash like two elements."""
        a = {"used_entities": ["a,b"]}
        b = {"used_entities": ["a", "b"]}
        assert compute_checksum(a) != compute_checksum(b)

    def test_list_element_boundary_collides_under_legacy(self):
        a = {"used_entities": ["a,b"]}
        b = {"used_entities": ["a", "b"]}
        assert compute_checksum_legacy(a) == compute_checksum_legacy(b)

    def test_list_order_is_significant(self):
        a = {"used_entities": ["x", "y"]}
        b = {"used_entities": ["y", "x"]}
        assert compute_checksum(a) != compute_checksum(b)

    def test_empty_list_and_absent_field_agree(self):
        """An absent list field defaults to an empty list, so both spellings
        hash the same — the collision to fix is between a comma inside one
        element and two elements, not between absent and empty."""
        a = {"used_entities": [], "entity_type": "t"}
        b = {"entity_type": "t"}
        assert compute_checksum(a) == compute_checksum(b)

    def test_dict_and_dataclass_agree(self):
        """v1 hashed `None` as `''` through the dict path and as `'None'`
        through the dataclass path. v2 normalises both."""
        entry = ProvenanceEntry(
            entity_id="e1",
            entity_type="entity",
            activity_id="extraction",
            source_document="DOI:10.1/x",
        )
        assert compute_checksum(entry) == compute_checksum(entry.to_dict())

    def test_checksum_is_stable(self):
        entry = {"entity_type": "entity", "activity_id": "a", "used_entities": ["u1"]}
        assert compute_checksum(entry) == compute_checksum(dict(entry))


class TestVersionDispatch:
    def test_new_entry_verifies_without_a_flag(self):
        entry = {"entity_type": "entity", "checksum_version": CHECKSUM_VERSION}
        entry["checksum"] = compute_checksum(entry)
        assert verify_checksum(entry) is True

    def test_legacy_entry_still_verifies(self):
        """A row written before the fix has no version and must keep verifying,
        otherwise every existing store reports a broken chain on upgrade."""
        entry = {"entity_type": "entity", "activity_id": "a"}
        entry["checksum"] = compute_checksum_legacy(entry)
        assert "checksum_version" not in entry
        assert verify_checksum(entry) is True

    def test_legacy_checksum_fails_once_stamped_v2(self):
        entry = {"entity_type": "entity", "activity_id": "a"}
        entry["checksum"] = compute_checksum_legacy(entry)
        entry["checksum_version"] = CHECKSUM_VERSION
        assert verify_checksum(entry) is False

    def test_missing_version_reads_as_legacy(self):
        assert checksum_version({}) == LEGACY_CHECKSUM_VERSION
        assert checksum_version({"checksum_version": None}) == LEGACY_CHECKSUM_VERSION
        assert checksum_version({"checksum_version": "2"}) == CHECKSUM_VERSION

    def test_no_checksum_is_not_valid(self):
        assert verify_checksum({"entity_type": "entity"}) is False

    def test_hand_assigned_checksum_verifies_on_a_dict(self):
        """The documented usage assigns the digest by hand, never passing
        through the manager's stamping path, so compute_checksum() has to
        record the version itself or verification reads it as v1 and rejects
        a digest it just produced."""
        entry = {"entity_type": "entity", "activity_id": "a"}
        entry["checksum"] = compute_checksum(entry)
        assert entry["checksum_version"] == CHECKSUM_VERSION
        assert verify_checksum(entry) is True

    def test_hand_assigned_checksum_verifies_on_a_dataclass(self):
        entry = ProvenanceEntry(
            entity_id="e1",
            entity_type="entity",
            activity_id="a",
            agent_id="agent",
            agent_type="human",
            source_document="doc",
        )
        entry.checksum = compute_checksum(entry)
        assert entry.checksum_version == CHECKSUM_VERSION
        assert verify_checksum(entry) is True

    def test_stamping_does_not_change_the_digest(self):
        entry = {"entity_type": "entity", "activity_id": "a"}
        first = compute_checksum(entry)
        second = compute_checksum(entry)
        assert first == second


class TestStorageRoundTrip:
    def test_checksum_version_survives_sqlite(self, tmp_path):
        db_path = str(tmp_path / "roundtrip.db")
        mgr = ProvenanceManager(storage_path=db_path)
        mgr.track_entity("E1", source="doc1")

        reloaded = ProvenanceManager(storage_path=db_path)
        entry = reloaded.storage.retrieve("E1")
        assert entry.checksum_version == CHECKSUM_VERSION
        assert verify_checksum(entry) is True

    def test_verify_chain_passes_on_a_fresh_store(self, tmp_path):
        db_path = str(tmp_path / "clean.db")
        mgr = ProvenanceManager(storage_path=db_path)
        mgr.track_entity("E1", source="doc1")
        mgr.track_entity("E2", source="doc1")
        assert mgr.verify_chain()["valid"] is True


class TestEndToEndDetection:
    def test_verify_chain_detects_a_char_moved_across_a_field_boundary(self, tmp_path):
        """The attack from issue #2136: move one character from the end of
        `entity_type` to the start of `activity_id`, leave the stored checksum
        alone, and the concatenation is byte-identical. v2 must notice."""
        db_path = str(tmp_path / "boundary.db")
        mgr = ProvenanceManager(storage_path=db_path)
        mgr.track_entity("E1", source="doc1")

        conn = sqlite3.connect(db_path)
        entity_type, activity_id = conn.execute(
            "SELECT entity_type, activity_id FROM provenance WHERE entity_id = 'E1'"
        ).fetchone()
        assert entity_type and activity_id
        conn.execute(
            "UPDATE provenance SET entity_type = ?, activity_id = ? WHERE entity_id = 'E1'",
            (entity_type[:-1], entity_type[-1] + activity_id),
        )
        conn.commit()
        conn.close()

        result = ProvenanceManager(storage_path=db_path).verify_chain()
        assert result["valid"] is False
        assert any(
            link["reason"] == "checksum_mismatch" for link in result["broken_links"]
        )

    def test_verify_chain_still_detects_a_plain_field_edit(self, tmp_path):
        db_path = str(tmp_path / "edit.db")
        mgr = ProvenanceManager(storage_path=db_path)
        mgr.track_entity("E1", source="doc1")

        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE provenance SET source_document = 'doc2' WHERE entity_id = 'E1'"
        )
        conn.commit()
        conn.close()

        result = ProvenanceManager(storage_path=db_path).verify_chain()
        assert result["valid"] is False
