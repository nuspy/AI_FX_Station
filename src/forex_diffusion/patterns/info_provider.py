from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import json

@dataclass
class PatternInfo:
    key: str
    name: str
    effect: str
    benchmarks: Dict[str, Any]
    bull: Dict[str, Any]
    bear: Dict[str, Any]
    image_resource: Optional[str] = None
    links: Optional[list] = None
    description: str = ""

class PatternInfoProvider:
    def __init__(self, json_path: Path) -> None:
        self.json_path = json_path
        self._db = {}
        try:
            self._db = json.loads(Path(json_path).read_text())
        except Exception:
            self._db = {}

    def describe(self, key: str) -> Optional[PatternInfo]:
        d = self._db.get(key, None)
        if not d:
            return None
        return PatternInfo(
            key=key,
            name=d.get("name", key),
            effect=d.get("effect",""),
            benchmarks=d.get("benchmarks",{}),
            bull=d.get("bull",{}),
            bear=d.get("bear",{}),
            image_resource=d.get("image",None),
            links=d.get("links",[]),
            description=d.get("description",""),
        )
