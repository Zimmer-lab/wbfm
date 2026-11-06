import numpy as np

from wbfm.utils.nwb.utils_nwb_export import _summary_ids_from_blobquant


def _make_blobquant(ids_per_blob):
    """Build a (n_blobs, n_times, 5) blobquant with IDs in column 4."""
    n_blobs = len(ids_per_blob)
    n_times = max(len(ids) for ids in ids_per_blob)
    blobquant = np.zeros((n_blobs, n_times, 5), dtype=object)
    for i, ids in enumerate(ids_per_blob):
        for t, ident in enumerate(ids):
            blobquant[i, t, 4] = ident
    return blobquant


def test_gap_at_start_keeps_id():
    """Neurons missing at frame 0 must still get their summary ID.

    Regression test: sampling frame 0 only gave '' here, which blanked
    almost all IDs for datasets with tracking gaps at the start
    (e.g. 2023-11-30_13-58_wt_worm3_FM kept 3 of 36 IDs).
    """
    blobquant = _make_blobquant([
        ['', '', 'SAAVL', 'SAAVL', 'SAAVL'],
        ['ALA', 'ALA', 'ALA', 'ALA', 'ALA'],
    ])
    assert list(_summary_ids_from_blobquant(blobquant)) == ['SAAVL', 'ALA']


def test_all_empty_stays_empty():
    blobquant = _make_blobquant([['', '', '0', 'nan']])
    assert list(_summary_ids_from_blobquant(blobquant)) == ['']


def test_mode_wins_on_conflict():
    blobquant = _make_blobquant([['AVAR', 'AVAL', 'AVAL', 'AVAL']])
    assert list(_summary_ids_from_blobquant(blobquant)) == ['AVAL']
