"""Persistent episodic memory for experience-based learning."""
import json
import os
import time
from copy import deepcopy


class EpisodicMemory:
    """Store structured experiences and retrieve relevant past episodes."""

    def __init__(self, path="data/episodes.json", capacity=10000):
        self.path = path
        self.capacity = max(1, int(capacity))
        self.episodes = []
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                self.episodes = data[-self.capacity:]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.episodes = []

    def _save(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.episodes[-self.capacity:], handle, ensure_ascii=False, indent=2)
        os.replace(temporary, self.path)

    def remember(self, event, metadata=None, **fields):
        episode = {
            "id": f"episode-{time.time_ns()}",
            "timestamp": time.time(),
            "event": deepcopy(event),
            "metadata": deepcopy(metadata or {}),
        }
        episode.update({key: deepcopy(value) for key, value in fields.items()})
        self.episodes.append(episode)
        self.episodes = self.episodes[-self.capacity:]
        self._save()
        return deepcopy(episode)

    def record(self, goal, observation=None, action=None, outcome=None,
               reflection=None, success=None, metadata=None):
        return self.remember(
            event={"goal": goal, "observation": observation, "action": action,
                   "outcome": outcome, "reflection": reflection, "success": success},
            metadata=metadata,
        )

    def recent(self, n=20):
        return deepcopy(self.episodes[-max(1, int(n)):])

    def search(self, query, limit=5):
        terms = set(str(query).lower().split())
        if not terms:
            return []
        scored = []
        for episode in self.episodes:
            text = json.dumps(episode, ensure_ascii=False).lower()
            score = sum(1 for term in terms if term in text)
            if score:
                scored.append((score, episode.get("timestamp", 0), episode))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [deepcopy(item[2]) for item in scored[:max(1, int(limit))]]

    def clear(self):
        self.episodes = []
        self._save()
