import sys
import types

import pytest

from jevcal.backends import LayaBackend, QwenRlcdBackend, create_backend


class FakeAgent:
    def predict(self, state, questions):
        assert state == "state"
        assert "q" in questions
        return {"answers": {"q": {"noul": 0.75}}}


def test_laya_backend_normalizes_metadata(monkeypatch):
    fake_laya = types.SimpleNamespace(load=lambda *args, **kwargs: FakeAgent())
    monkeypatch.setitem(sys.modules, "laya", fake_laya)
    backend = LayaBackend("example/model", device="cpu")
    result = backend.ask("state", {"q": {"type": "noul", "instructions": "true?"}})
    assert result["answers"]["q"]["noul"] == 0.75
    assert result["model"] == "example/model"
    assert result["usage"] == {}


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        create_backend("unknown")


def test_qwen_backend_normalizes_metadata(monkeypatch):
    class FakeModel:
        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def score(self, *, state, questions):
            return {"model": "bouncy-2.0.0", "answers": {"q": {"noul": 0.6}}}

    fake_auto = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr("transformers.AutoModel", fake_auto)
    backend = QwenRlcdBackend("example/qwen", device="cpu")
    result = backend.ask("state", {"q": {"type": "noul", "instructions": "true?"}})
    assert result["model"] == "example/qwen"
    assert result["backend_model"] == "bouncy-2.0.0"


def test_eve_rlcd_backend_maps_descriptions_back_to_keys():
    from jevcal.backends import EveRlcdBackend

    class FakeChoice:
        def __init__(self, question, options):
            self.question, self.options = question, options

    class FakeNoul:
        def __init__(self, question):
            self.question = question

    class FakeDecider:
        def ask(self, state, prims):
            out = []
            for p in prims:
                if isinstance(p, FakeNoul):
                    out.append({"kind": "noul", "p_true": 0.3})
                else:
                    probs = {d: (0.8 if i == 1 else 0.2) for i, d in enumerate(p.options)}
                    out.append({"kind": "choice", "value": p.options[1], "probs": probs,
                                "confidence": 0.8})
            return out

    b = EveRlcdBackend.__new__(EveRlcdBackend)
    b._choice, b._noul, b._decider, b.name = FakeChoice, FakeNoul, FakeDecider(), "fake"
    r = b.ask("s", {"q": {"type": "choice", "instructions": "i",
                          "criteria": {"heads": "H up", "tails": "T up"}},
                    "n": {"type": "noul", "instructions": "yes?"}})
    assert r["answers"]["q"]["choice"] == "tails"
    assert r["answers"]["q"]["probabilities"] == {"heads": 0.2, "tails": 0.8}
    assert r["answers"]["n"] == {"type": "noul", "noul": 0.3}
