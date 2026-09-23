"""Decision-model backends with a shared Jev-compatible response contract."""

from __future__ import annotations

import os
from typing import Protocol

from jevcal import client

QWEN_RLCD_REVISION = "b5b99a9fc2422870881e923edd478d9dc84924d6"


class Backend(Protocol):
    name: str
    parallel_safe: bool

    def ask(self, state: str, questions: dict) -> dict: ...


class JevBackend:
    """Remote TypeSafe backend used by the original experiments."""

    name = "jev-latest"
    parallel_safe = True

    def __init__(self, model: str = client.MODEL):
        self.name = model
        self._key = client.load_key()

    def ask(self, state: str, questions: dict) -> dict:
        return client.ask(self._key, state, questions, model=self.name)


class LayaBackend:
    """Local open-weight Laya backend.

    Importing Laya is deliberately delayed so offline tests do not require torch.
    Hugging Face honors HF_HOME, so callers can place downloads in a disposable
    directory and remove it after an experiment.
    """

    parallel_safe = False

    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        *,
        device: str = "cpu",
        subfolder: str | None = None,
        hf_token: str | None = None,
    ):
        os.environ.setdefault("USE_TF", "0")
        try:
            import laya
        except ImportError as exc:
            raise RuntimeError("Install the local backend with: venv/bin/pip install laya") from exc
        self.name = f"{model}/{subfolder}" if subfolder else model
        self._agent = laya.load(model, device=device, token=hf_token, subfolder=subfolder)

    def ask(self, state: str, questions: dict) -> dict:
        result = self._agent.predict(state, questions)
        # Laya already uses the same `answers` schema. Add stable metadata so
        # existing experiment writers and analyzers can consume it unchanged.
        if reported := result.get("model"):
            result["backend_model"] = reported
        result["model"] = self.name
        result.setdefault("usage", {})
        return result


class QwenRlcdBackend:
    """Local Qwen3 option-scoring checkpoint with its published remote-code wrapper."""

    parallel_safe = False

    def __init__(
        self,
        model: str = "thefloydd/qwen3-0.6b-rlcd",
        *,
        device: str = "cpu",
        hf_token: str | None = None,
    ):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Install the local backend with: venv/bin/pip install transformers torch"
            ) from exc
        self.name = model
        self._model = AutoModel.from_pretrained(
            model,
            trust_remote_code=True,
            dtype="auto",
            token=hf_token,
            revision=QWEN_RLCD_REVISION,
        ).to(device).eval()

    def ask(self, state: str, questions: dict) -> dict:
        result = self._model.score(state=state, questions=questions)
        if reported := result.get("model"):
            result["backend_model"] = reported
        result["model"] = self.name
        return result


EVE_RLCD_REPO = client.ROOT / "data" / "external" / "eve-rlcd"  # commit 57a179b7
EVE_RLCD_MODEL = client.ROOT / "data" / "external" / "models" / "qwen3-0.6b-rlcd-decision"
EVE_RLCD_REVISION = "b327ec5efb5fdbf8bfafa3b369720ac5f6434b05"


class EveRlcdBackend:
    """anthonym21/qwen3-0.6b-rlcd-decision through its author's own `rlcd.decide.Decider`.

    The only open checkpoint we found trained with actual REINFORCE against a proper
    scoring rule (reward c - p_a). The loader verifies the export's sha256 digests and
    loads a stock Qwen3 body, with no remote code. Its prompt shows option *descriptions*
    as lettered lines, never our keys, so key-name manipulations are a no-op for this model
    and only order and description effects can be measured.
    """

    parallel_safe = False

    def __init__(self, model: str | None = None, *, device: str = "mps"):
        import sys

        if str(EVE_RLCD_REPO) not in sys.path:
            sys.path.insert(0, str(EVE_RLCD_REPO))
        from rlcd.decide import ChoiceQ, Decider, NoulQ

        self._choice, self._noul = ChoiceQ, NoulQ
        self.name = f"anthonym21/qwen3-0.6b-rlcd-decision@{EVE_RLCD_REVISION[:8]}"
        self._decider = Decider.load(str(model or EVE_RLCD_MODEL), device=device)

    def ask(self, state: str, questions: dict) -> dict:
        names, prims, keymaps = [], [], []
        for name, q in questions.items():
            names.append(name)
            if q["type"] == "choice":
                keys = list(q["criteria"])
                descs = [q["criteria"][k] or k for k in keys]
                prims.append(self._choice(q["instructions"], descs))
                keymaps.append(dict(zip(descs, keys, strict=True)))
            elif q["type"] == "noul":
                prims.append(self._noul(q["instructions"]))
                keymaps.append(None)
            else:
                raise ValueError(f"unsupported question type for eve-rlcd: {q['type']}")
        answers = {}
        for name, raw, keymap in zip(names, self._decider.ask(state, prims), keymaps, strict=True):
            if keymap is None:
                answers[name] = {"type": "noul", "noul": raw["p_true"]}
            else:
                probs = {keymap[d]: p for d, p in raw["probs"].items()}
                answers[name] = {"type": "choice", "choice": keymap[raw["value"]],
                                 "probabilities": probs, "confidence": raw["confidence"]}
        return {"model": self.name, "answers": answers, "usage": {}}


def create_backend(
    kind: str,
    *,
    model: str | None = None,
    device: str = "cpu",
    subfolder: str | None = None,
) -> Backend:
    """Construct a supported backend without exposing implementation details."""
    if kind == "jev":
        return JevBackend(model or client.MODEL)
    if kind == "laya":
        return LayaBackend(
            model or "convaiinnovations/laya",
            device=device,
            subfolder=subfolder,
            hf_token=os.environ.get("HF_TOKEN"),
        )
    if kind == "qwen-rlcd":
        return QwenRlcdBackend(
            model or "thefloydd/qwen3-0.6b-rlcd",
            device=device,
            hf_token=os.environ.get("HF_TOKEN"),
        )
    if kind == "eve-rlcd":
        return EveRlcdBackend(model, device=device if device != "cpu" else "mps")
    raise ValueError(f"unknown backend: {kind}")
