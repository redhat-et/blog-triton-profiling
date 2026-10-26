#!/usr/bin/env python3
"""Smoke-test the LLM provider configured for KernelAgent.

Checks that the model id in OPENAI_MODEL routes to the provider its
backend inference selects (claude-* ids mean the Anthropic API unless
OPENAI_BASE_URL points at an OpenAI-compatible server), that the
provider is available (API key present), and that a minimal completion
round-trips. Prints VERIFY_OK on success. KA_REASONING_EFFORT is
validated and echoed before the round-trip, so a typo in the knob fails
this cheap smoke test instead of a long run.
"""

from __future__ import annotations

import os
import sys

from ka_common import ANTHROPIC_DEFAULT_MODEL, infer_llm_backend


def _hint(backend: str) -> str:
    if backend == "anthropic":
        return "hint: export ANTHROPIC_API_KEY"
    return (
        "hint: export OPENAI_API_KEY (any non-empty value works for local\n"
        "      OpenAI-compatible servers, e.g. OPENAI_API_KEY=local)"
    )


def main() -> int:
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not model:
        print("ERROR: OPENAI_MODEL is not set (OPENAI_MODEL on the make "
              "command line)", file=sys.stderr)
        print(
            "  Anthropic API:   OPENAI_MODEL=claude-sonnet-4-6 "
            f"(or another claude-* id; default {ANTHROPIC_DEFAULT_MODEL})",
            file=sys.stderr,
        )
        print(
            "  OpenAI API or a local OpenAI-compatible server (selected via\n"
            "                   OPENAI_BASE_URL): OPENAI_MODEL=<model id>",
            file=sys.stderr,
        )
        return 1

    from utils.providers import get_model_provider
    from utils.providers.base import reasoning_effort_from_env

    # The entrypoint derives LLM_BACKEND from the model id; honor it when
    # present (it also carries a manual override), infer otherwise.
    backend = os.environ.get("LLM_BACKEND") or infer_llm_backend(model)

    try:
        provider = get_model_provider(model)
    except ValueError as exc:
        print(
            f"ERROR: no provider for model '{model}': {exc}", file=sys.stderr
        )
        print(_hint(backend), file=sys.stderr)
        return 1

    print(
        f"provider={provider.name} model={model} "
        f"available={provider.is_available()}"
    )
    if provider.name != backend:
        print(
            f"ERROR: expected the {backend} provider for model '{model}', "
            f"got '{provider.name}'",
            file=sys.stderr,
        )
        if backend == "openai":
            print(
                "hint: claude-* ids route to the Anthropic API unless\n"
                "      OPENAI_BASE_URL points at an OpenAI-compatible\n"
                "      server\n",
                file=sys.stderr,
            )
        return 1
    if not provider.is_available():
        print(
            f"ERROR: the {provider.name} provider is not available",
            file=sys.stderr,
        )
        print(_hint(backend), file=sys.stderr)
        return 1

    try:
        effort = reasoning_effort_from_env()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"reasoning_effort={effort or '<model default>'}")

    response = provider.get_response(
        model, [{"role": "user", "content": "Reply with exactly: OK"}]
    )
    content = (response.content or "").strip()
    print(f"smoke: {content[:200]!r}")
    if not content:
        print("ERROR: empty LLM response", file=sys.stderr)
        return 1
    print("VERIFY_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
