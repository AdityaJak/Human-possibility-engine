"""External-dataset acquisition with SHA-256 verification (namespace: external_dataset).

Raw external data is NEVER committed. Files are downloaded on demand into the gitignored
``datasets/external/`` directory (or read from a path given by an environment variable) and
verified against the hashes recorded below, which were observed on 2026-11-04.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_DIR = REPO_ROOT / "datasets" / "external"


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    url: str
    sha256: str | None
    size_bytes: int | None
    licence: str
    licence_evidence: str
    citation: str
    retrieved: str
    env_var: str


PSYCH101_FULL = DatasetSource(
    name="psych101_full",
    url="https://huggingface.co/datasets/marcelbinz/Psych-101/resolve/main/prompts_training.jsonl",
    sha256="7a370a94084acd6f73f20d6559214683cfadcab8a1f69e369df285ba75ccef01",
    size_bytes=859_179_475,
    licence="Apache-2.0",
    licence_evidence="Hugging Face dataset card metadata 'license: apache-2.0' and 'Licensing Information: Apache License 2.0'",
    citation="Binz, M. et al. (2024). Centaur: a foundation model of human cognition. arXiv:2410.20268",
    retrieved="2026-11-04",
    env_var="HPE_PSYCH101_FULL",
)

# Deterministic line-order-preserving filter of PSYCH101_FULL to LEARNING_EXPERIMENT_PREFIXES.
PSYCH101_LEARNING_SUBSET = DatasetSource(
    name="psych101_learning_subset",
    url="derived: filter_jsonl(PSYCH101_FULL, LEARNING_EXPERIMENT_PREFIXES)",
    sha256="9b085cfbd5506162ae4925b5ca05b3fcb76a69b885395b126d8aa78cf2330297",
    size_bytes=57_296_364,
    licence="Apache-2.0 (inherited from Psych-101)",
    licence_evidence="derived from PSYCH101_FULL",
    citation=PSYCH101_FULL.citation,
    retrieved="2026-11-04",
    env_var="HPE_PSYCH101_SUBSET",
)

LEARNING_EXPERIMENT_PREFIXES: tuple[str, ...] = (
    "steingroever2015data/",
    "collsiöö2023MCPL/",
    "speekenbrink2008learning/",
    "gershman2018deconstructing/",
    "wilson2014humans/",
    "lefebvre2017behavioural/",
    "garcia2023experiential/",
    "plonsky2018when/",
    "waltz2020differential/",
    "gershman2020reward/",
)

HEARTSTEPS_SUGGESTIONS = DatasetSource(
    name="heartsteps_v1_suggestions",
    url="https://raw.githubusercontent.com/klasnja/HeartStepsV1/main/data_files/suggestions.csv",
    sha256="60ee7896183d7084a084f9a6e8ef2d0afea2e7fecdb41d6b15382ce1745c5ddb",
    size_bytes=4_371_854,
    licence="CC-BY-4.0",
    licence_evidence="LICENSE file in github.com/klasnja/HeartStepsV1 is 'Attribution 4.0 International'",
    citation=(
        "Klasnja, P., Smith, S., Seewald, N. J., Lee, A., Hall, K., Luers, B., Hekler, E. B., & Murphy, S. A. "
        "(2019). Efficacy of Contextually Tailored Suggestions for Physical Activity: A Micro-randomized "
        "Optimization Trial of HeartSteps. Annals of Behavioral Medicine, 53(6), 573-582. doi:10.1093/abm/kay067"
    ),
    retrieved="2026-11-04",
    env_var="HPE_HEARTSTEPS_SUGGESTIONS",
)


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset is neither cached nor downloadable."""


class HashMismatch(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(path: Path, expected_sha256: str | None) -> str:
    """Return the file's SHA-256; raise ``HashMismatch`` if it differs from ``expected_sha256``."""
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise HashMismatch(f"{path}: sha256 {digest} != expected {expected_sha256}")
    return digest


def download(url: str, dest: Path, expected_sha256: str | None = None, timeout: float = 600.0) -> str:
    """Stream ``url`` to ``dest`` (atomic rename) and verify its hash."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
        while True:
            b = resp.read(1 << 20)
            if not b:
                break
            h.update(b)
            out.write(b)
    digest = h.hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise HashMismatch(f"{url}: sha256 {digest} != expected {expected_sha256}")
    tmp.replace(dest)
    return digest


def filter_jsonl(src: Path, dest: Path, prefixes: Iterable[str], key: str = "experiment") -> int:
    """Copy JSONL lines whose ``key`` starts with any prefix, preserving order and bytes."""
    pref = tuple(prefixes)
    n = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as f, open(dest, "w", encoding="utf-8") as o:
        for line in f:
            if json.loads(line)[key].startswith(pref):
                o.write(line)
                n += 1
    return n


def default_path(source: DatasetSource) -> Path:
    suffix = ".csv" if source.url.endswith(".csv") else ".jsonl"
    return EXTERNAL_DIR / f"{source.name}{suffix}"


def locate(source: DatasetSource, *, verify_hash: bool = True) -> Path:
    """Find a local copy (env var first, then ``datasets/external``). Never downloads."""
    candidates = []
    env = os.environ.get(source.env_var)
    if env:
        candidates.append(Path(env))
    candidates.append(default_path(source))
    for c in candidates:
        if c.exists():
            if verify_hash:
                verify(c, source.sha256)
            return c
    raise DatasetUnavailable(
        f"{source.name} not found (set {source.env_var} or run datasets/adapters/ to download it)"
    )


def provenance(source: DatasetSource, path: Path | None = None) -> dict[str, str | int | None]:
    return {
        "name": source.name,
        "url": source.url,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
        "licence": source.licence,
        "licence_evidence": source.licence_evidence,
        "citation": source.citation,
        "retrieved": source.retrieved,
        "local_path": str(path) if path else None,
    }
