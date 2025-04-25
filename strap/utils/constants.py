import os

# Base directories - can be overridden with environment variables
BASE_DATA_DIR = os.environ.get("STRAP_BASE_DATA_DIR", os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data"))
EMBEDDED_DATA_DIR = os.environ.get("STRAP_EMBEDDED_DATA_DIR", os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data/embedded"))
RETRIEVAL_RESULTS_DIR = os.environ.get("STRAP_RETRIEVAL_DIR", os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data/retrieved"))

# Default repo root (kept for backward compatibility)
REPO_ROOT = os.environ.get("STRAP_DATA_DIR", os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
))
