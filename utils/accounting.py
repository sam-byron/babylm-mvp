# --- filename: utils/accounting.py
import json
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExposureAccounting:
    """Tracks whitespace-separated *word* exposures and triggers checkpointing.
    """
    milestones: List[int]
    max_words_seen: int
    save_dir: str
    words_seen: int = 0
    last_saved_idx: int = -1

    def add(self, words: int) -> List[int]:
        """Add exposures, return list of milestone indices crossed since last call."""
        prev = self.words_seen
        self.words_seen += int(words)
        crossed = []
        for i, m in enumerate(self.milestones):
            if prev < m <= self.words_seen:
                crossed.append(i)
        if self.words_seen >= self.max_words_seen:
            print(f"[accounting] Reached max_words_seen={self.max_words_seen}")
        return crossed

    def should_stop(self) -> bool:
        return self.words_seen >= self.max_words_seen

    def dump_state(self):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "exposure.json"), "w") as f:
            json.dump({"words_seen": self.words_seen, "milestones": self.milestones, "max": self.max_words_seen}, f)
