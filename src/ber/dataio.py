"""Reading the challenge TSVs and writing submission files.

Every file in this challenge is tab-separated and addresses contain commas, so the
separator is never negotiable - reading one of these as CSV silently yields a single
column. Files are streamed line by line throughout: the three test sources alone are
1.2 GB and the machine this runs on has ~20 GB of free disk and 23 GB of RAM.
"""

import os

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def read_source(path, country=None, keep=None):
    """Yield (entity_id, name, address, country) from a source TSV.

    ``country`` restricts to one country label (blocking is country-partitioned, so we only
    ever want one shard in memory at a time). ``keep`` is an optional predicate on the
    entity id, used to take a validation subsample.
    """
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if tuple(header) != SOURCE_COLUMNS:
            raise ValueError(f"{path}: unexpected header {header}")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            if country is not None and parts[3] != country:
                continue
            if keep is not None and not keep(parts[0]):
                continue
            yield parts[0], parts[1], parts[2], parts[3]


def read_ground_truth(path, keep=None):
    """Return {source1_entity_id: set(matched ids)}. Empty set means a singleton."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts[0]:
                continue
            if keep is not None and not keep(parts[0]):
                continue
            ids = parts[1] if len(parts) > 1 else ""
            out[parts[0]] = {x for x in ids.split(",") if x}
    return out


def country_counts(path):
    """Country label -> record count, for planning shard sizes."""
    counts = {}
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            counts[parts[3]] = counts.get(parts[3], 0) + 1
    return counts


def write_id_lists(path, rows, header):
    """Write a submission-style TSV: one row per S1 entity, comma-joined id list.

    ``rows`` yields (source1_entity_id, iterable_of_ids). Ids are written unquoted and
    comma-separated, exactly as the challenge specifies, and an entity with no ids gets an
    empty second field rather than being omitted - a missing row causes rejection.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        for eid, ids in rows:
            fh.write(eid)
            fh.write("\t")
            if ids:
                fh.write(",".join(ids))
            fh.write("\n")
            n += 1
    return n
