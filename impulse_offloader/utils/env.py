"""Environment helpers with optional python-dotenv support."""
from __future__ import annotations

import importlib.util
from typing import Any

if importlib.util.find_spec("dotenv"):
    from dotenv import load_dotenv as _load_dotenv  # type: ignore[attr-defined]
else:
    def _load_dotenv(*args: Any, **kwargs: Any) -> bool:  # pragma: no cover - fallback
        """Fallback loader when python-dotenv is unavailable."""
        return False


def load_environment(*args: Any, **kwargs: Any) -> Any:
    """Load environment variables from a .env file if supported."""
    return _load_dotenv(*args, **kwargs)
