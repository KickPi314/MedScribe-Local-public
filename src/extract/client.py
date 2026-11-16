"""Minimal Ollama HTTP client (Phase 3, spec section 3.2).

Written against the Python standard library (``urllib.request`` + ``json``) and
nothing else. That is a deliberate choice, not laziness:

* ``requirements.txt`` is deliberately minimal, and the whole job here is one
  POST of a JSON body containing a base64 image to a loopback address. ``urllib``
  does that in about thirty lines. Adding ``requests`` or the ``ollama`` package
  would add a dependency tree to audit for outbound calls (spec section 7:
  "audit the dependency tree for outbound calls") in exchange for nothing this
  stage needs -- no retries with backoff, no connection pooling, no streaming.
* a thin client keeps the failure modes legible. Every error below names the
  URL, the model and the thing the operator should do about it.

NETWORK POSTURE (spec section 1.1). This module is the only place in the
codebase that opens a socket. It refuses to connect to anything but a loopback
host from ``[ollama].allowed_hosts`` -- validated in
:func:`src.extract.config.load_extract_config` and re-checked here, because a
``Config`` can also be built by ``--set`` overrides which bypass the loader's
validation. A loopback request cannot leave the machine, so the app still
behaves correctly with the network adapter disabled, which is the test spec
section 1.1 actually sets.

"Never guess" (spec section 1.3) applies to transport too: a timeout, a refused
connection or an HTTP error raises. It never returns an empty-but-successful
result that a caller might mistake for "the model read nothing on this page".
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..preprocess.config import Config, ConfigError
from .config import model_options, ollama_base_url


class OllamaError(RuntimeError):
    """Ollama is unreachable, unhealthy, or returned something unusable."""


@dataclass
class ModelReply:
    """One completed inference call.

    ``text`` is the payload the parser will work on; ``text_field`` records
    which JSON key of the reply it came from, because on the verified local
    stack that is *not* always ``response`` -- see ``[model].response_fields``
    in config/extract.toml for the measurement. Keeping the source visible means
    the workaround shows up in the metrics output instead of hiding in a
    fallback chain.
    """

    text: str
    text_field: str
    raw: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def done_reason(self) -> str | None:
        r = self.raw.get("done_reason")
        return None if r is None else str(r)

    @property
    def truncated(self) -> bool:
        """Output was cut off by ``[model.options].num_predict``.

        A truncated reply is treated as a failed read by the parser, not as a
        partial answer to be completed with defaults.
        """
        return self.done_reason == "length"

    def meta(self) -> dict[str, Any]:
        """The token/timing counters, for the metrics output. No payload text."""
        keys = (
            "model",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
        return {k: self.raw.get(k) for k in keys if k in self.raw}


class OllamaClient:
    """A single-purpose client for one local Ollama instance."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.base_url = ollama_base_url(cfg)
        self.model = str(cfg.get("model", "name"))
        self.endpoint = str(cfg.get("ollama", "endpoint"))
        self.timeout_s = float(cfg.get("ollama", "timeout_s"))
        self.keep_alive = cfg.get("ollama", "keep_alive")
        self.response_fields = [str(f) for f in cfg.get("model", "response_fields")]
        self._format = str(cfg.get("model", "format"))
        self._think = bool(cfg.get("model", "think"))
        self._prefill = str(cfg.get_opt("model", "assistant_prefill", ""))
        self._options = model_options(cfg)

        # Re-check the loopback guard here as well as in the config loader: a
        # Config built through --set overrides never passes through the loader's
        # validation, and this is the module that would do the connecting.
        host = str(cfg.get("ollama", "host"))
        allowed = [str(h) for h in cfg.get("ollama", "allowed_hosts")]
        if host not in allowed:
            raise ConfigError(
                f"refusing to connect to {host!r}: not in [ollama].allowed_hosts "
                f"{allowed}. Spec section 1.1 (zero network at runtime) limits this "
                f"client to a loopback address."
            )

    # -- transport -------------------------------------------------------
    def _post(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # pragma: no cover - best effort only
                pass
            raise OllamaError(
                f"{url} returned HTTP {exc.code}: {detail or exc.reason}. Model "
                f"requested: {self.model!r}."
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {url} ({exc.reason}). Ollama is a "
                f"documented local prerequisite (spec section 3.2): start it with "
                f"`ollama serve`, then check `curl {self.base_url}/api/version`. "
                f"No value is extracted and nothing is guessed for this page."
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"{url} did not respond within [ollama].timeout_s="
                f"{self.timeout_s:g}s. A cold model load on CPU took 216s on the "
                f"reference machine; raise the timeout if the model is being "
                f"loaded from disk on every call, or set [ollama].keep_alive to "
                f"hold it resident."
            ) from exc
        elapsed = time.perf_counter() - t0
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OllamaError(
                f"{url} returned a body that is not JSON ({exc}): {payload[:300]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise OllamaError(f"{url} returned {type(parsed).__name__}, expected a JSON object")
        return parsed, elapsed

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {url} ({exc.reason}). Start it with "
                f"`ollama serve`."
            ) from exc

    # -- health ----------------------------------------------------------
    def version(self) -> str:
        return str(self._get_json("/api/version").get("version", "unknown"))

    def available_models(self) -> list[str]:
        data = self._get_json("/api/tags")
        return [str(m.get("name")) for m in data.get("models", []) if m.get("name")]

    def preflight(self) -> list[str]:
        """Check the server is up and the configured model is pulled.

        Returns informational notes. Raises :class:`OllamaError` when the server
        is unreachable or the model is missing -- both are operator-fixable setup
        problems, and failing here produces a far clearer message than a 404 from
        the middle of a batch. Model downloads happen out-of-band during setup
        (spec section 1.1), so this never tries to pull anything.
        """
        notes = [f"ollama {self.version()} at {self.base_url}"]
        if not bool(self.cfg.get("ollama", "preflight_model_check")):
            notes.append(
                "[ollama].preflight_model_check is false; model presence not verified"
            )
            return notes
        models = self.available_models()
        if self.model not in models:
            raise OllamaError(
                f"model {self.model!r} is not available on the local Ollama "
                f"instance. Pulled models: {models or ['(none)']}. Pull it "
                f"out-of-band during setup (`ollama pull {self.model}`) -- spec "
                f"section 1.1 allows model downloads at setup time only, and this "
                f"code never downloads anything."
            )
        notes.append(f"model {self.model!r} present ({len(models)} model(s) pulled)")
        return notes

    # -- inference -------------------------------------------------------
    def generate(self, prompt: str, image_bytes: bytes,
                 options: dict[str, Any] | None = None) -> ModelReply:
        """One narrow, single-purpose call: one prompt, one region image.

        Whole pages are never sent (spec section 3.4) -- the caller passes a
        region crop from Phase 2.

        ``options`` replaces ``[model.options]`` for this call. Phase 4 uses it
        for the per-region ``num_ctx`` (see
        :func:`src.extract.config.region_model_options`); it is still a config
        value, resolved by the caller, never a default invented here.
        """
        b64 = base64.b64encode(image_bytes).decode("ascii")
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,  # one region, one answer; nothing to stream to
            "think": self._think,
            "options": dict(self._options if options is None else options),
        }
        if self._format:
            body["format"] = self._format
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        if self.endpoint == "chat":
            path = "/api/chat"
            body["messages"] = [{"role": "user", "content": prompt, "images": [b64]}]
            if self._prefill:
                # See [model].assistant_prefill: an empty think block the model
                # continues from, so it answers instead of reasoning.
                body["messages"].append({"role": "assistant", "content": self._prefill})
        else:
            path = "/api/generate"
            body["prompt"] = prompt
            body["images"] = [b64]

        raw, elapsed = self._post(path, body)
        # /api/chat nests the reply under "message"; /api/generate is flat. Look
        # for the configured payload fields in whichever envelope came back.
        envelope = raw.get("message") if self.endpoint == "chat" else raw
        if not isinstance(envelope, dict):
            raise OllamaError(
                f"{self.base_url}{path} reply has no usable envelope: keys "
                f"{sorted(raw)}"
            )

        warnings: list[str] = []
        text = ""
        source = ""
        for i, name in enumerate(self.response_fields):
            # /api/chat calls the answer field "content"; /api/generate calls it
            # "response". Accept the generate spelling for both.
            key = "content" if self.endpoint == "chat" and name == "response" else name
            candidate = envelope.get(key)
            if isinstance(candidate, str) and candidate.strip():
                text, source = candidate, name
                if i > 0:
                    warnings.append(
                        f"reply payload came from the {name!r} field, not "
                        f"{self.response_fields[0]!r} (which was empty). See "
                        f"[model].response_fields in the extract config -- this is "
                        f"a known Ollama/qwen3-vl behaviour, recorded rather than "
                        f"hidden."
                    )
                break
        if not source:
            warnings.append(
                f"none of [model].response_fields {self.response_fields} held any "
                f"text; reply fields present: "
                f"{sorted(k for k, v in envelope.items() if isinstance(v, str))}"
            )
            source = "(none)"

        reply = ModelReply(
            text=text, text_field=source, raw=raw, duration_s=elapsed, warnings=warnings
        )
        if reply.truncated:
            effective = self._options if options is None else options
            reply.warnings.append(
                f"generation stopped on done_reason='length', i.e. it hit "
                f"num_predict={effective.get('num_predict', '(unset)')}. The reply is "
                f"incomplete and is treated as a failed read, not a partial answer."
            )
        return reply
