from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Protocol


@dataclass(frozen=True)
class TrialArtifact:
    results_root: Path
    scenario: str
    run_dir: Path
    result_path: Path
    run_tag: str
    seed: str
    result_file: str


class ResultsAdapter(Protocol):
    name: str

    def discover_trials(self, results_root: Path) -> Iterable[TrialArtifact]:
        ...

    def load_trial_object(self, artifact: TrialArtifact) -> Any:
        ...

    def extract_row(
        self,
        artifact: TrialArtifact,
        trial_object: Any,
        *,
        include_all_config: bool,
    ) -> Dict[str, Any]:
        ...
