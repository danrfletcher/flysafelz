"""Scoring for the NOTAM reader eval: harmful downgrades (G13) and per-field accuracy (G14), offline and model-free.

    labels      Dan's labels, and which are complete, partial, stale or unknown (E13)
    spots       the places and the hour a NOTAM is judged at (R3.4, R3.11)
    downgrade   score_item_run: harmful downgrades and over-caution against a label
    fields      field_accuracy and aggregate
    items       the sheet (evals/reader/items.yaml) and the NOTAM each item names

Nothing here calls a model. PR-8 runs the reader and calls these functions.
"""
