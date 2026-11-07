"""The NOTAM reader: one model call per NOTAM that fills the semantic fields of a NotamReading.

The only package that imports a model client. Code builds every coded field (id, area, schedule); the model fills only
what ReaderOutput names. See prompt.py for what the model is sent and read.py for the call.
"""
