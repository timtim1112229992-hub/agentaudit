"""Decision-level auditing of automated support policies in staged classroom tasks.

The package answers two questions that are usually merged. Does an automated agent
adjust its support to the state of a learner's work, and does it withdraw that
support over time? The two are estimated separately, because a system can do the
first perfectly while never doing the second.
"""
from __future__ import annotations

__all__ = ["config", "ingest", "derive", "lexicon", "recode", "indices", "sequence",
           "models", "resample", "figures", "provenance", "pipeline"]
__version__ = "0.1.0"
