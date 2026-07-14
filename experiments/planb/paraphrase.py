"""
Plan B — LLM-paraphrase the grounded descriptions for real lexical diversity.

The templated descriptions (even v2's larger bank) are still combinatorial
templates; the model can memorize their structure. This rewrites each description
with Claude into natural, varied prose — while staying strictly grounded: every
paraphrase must name exactly the degraded axes (with their severity) and nothing
else. We deduplicate by *degradation profile* (the set of degraded axes + their
severities + noise type), so 4,300 clips collapse to ~740 unique profiles → one
API call per profile returning a pool of variants, sampled per clip.

Grounding is verified programmatically: a paraphrase is kept only if it mentions a
keyword for every degraded axis and no keyword for any clean axis. Clips whose pool
ends up empty fall back to the v2 template (so output is always complete).

Requires ANTHROPIC_API_KEY in the environment (add it to ~/.zshrc; never paste a
key into a tracked file). Uses the official anthropic SDK + structured outputs.

Usage:
  python -m experiments.planb.paraphrase \
      --in experiments/results/planb/corpus_train_v2.jsonl \
      --out experiments/results/planb/corpus_train_v3.jsonl \
      --cache experiments/results/planb/paraphrase_pool.json [--model claude-opus-4-8]
"""

import argparse
import ast
import json
import os

import numpy as np

from experiments.planb import targets as T

SEV_WORD = {4: "slight", 3: "moderate", 2: "strong", 1: "severe"}
AXIS_NAME = {
    "noise": "background noise", "reverberation": "reverberation / room echo",
    "bandwidth": "band-limiting (a muffled, narrow-band tone)", "clipping": "clipping distortion",
    "discontinuity": "dropouts / choppiness", "loudness": "an inappropriate loudness level",
}
# keywords for grounding verification (per dimension)
KW = {
    "noise": ["noise", "hiss", "background", "static", "noisy", "music", "hum"],
    "reverberation": ["reverb", "echo", "room", "hall", "distant", "cavern", "reverberant", "ambien"],
    "bandwidth": ["muffl", "band", "narrow", "telephone", "dull", "tinny", "high frequenc",
                  "high-frequenc", "treble", "limited", "coloration", "boxy"],
    "clipping": ["clip", "distort", "crackl", "harsh", "saturat", "overload", "break up", "broken up"],
    "discontinuity": ["dropout", "drop-out", "gap", "chopp", "stutter", "discontin", "interrupt", "cut out"],
    "loudness": ["loud", "quiet", "level", "volume", "soft", "faint recording"],
}


def profile_key(rec):
    """Description-relevant facts: degraded axes + severity + noise type. Order
    canonical so clips that should read the same map to the same key."""
    nt = (rec["params"].get("noise") or {}).get("noise_type")
    return tuple(sorted((d, s, (nt if d == "noise" else None))
                        for d, s in rec["scores"].items() if s < 5))


def describe_profile(key):
    """Human-readable problem list for the prompt."""
    if not key:
        return None
    items = []
    for dim, sev, nt in key:
        name = "background music" if (dim == "noise" and nt == "music") else AXIS_NAME[dim]
        items.append(f"{SEV_WORD[sev]} {name}")
    return items


def verify(text, key):
    """Keep a paraphrase only if it names every degraded axis and no clean axis."""
    t = text.lower()
    degraded = {dim for dim, _, _ in key}
    for dim in degraded:
        if not any(k in t for k in KW[dim]):
            return False
    for dim in set(KW) - degraded:
        # don't reject on incidental words shared with a degraded axis sense
        if any(k in t for k in KW[dim]):
            return False
    return True


def gen_pool(client, model, key, n=6):
    """One API call -> up to n diverse, grounded paraphrases for this profile."""
    if not key:
        prompt = (f"Write {n} diverse, natural one-sentence descriptions of a CLEAN, "
                  "high-quality speech recording with no audible problems. Vary the wording.")
    else:
        problems = "; ".join(describe_profile(key))
        prompt = (
            "You write short, natural descriptions of speech-recording quality problems "
            "for a training dataset.\n\n"
            f"This recording has EXACTLY these problems: {problems}.\n\n"
            f"Write {n} diverse descriptions (one or two sentences each). Rules:\n"
            "- Each description MUST mention every listed problem.\n"
            "- Do NOT mention any problem that is not listed.\n"
            "- Vary the wording, sentence structure, and the order you mention problems.\n"
            "- Natural prose only: no scores, numbers, bullet points, or quotation marks."
        )
    resp = client.messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": {
            "type": "object",
            "properties": {"descriptions": {"type": "array", "items": {"type": "string"}}},
            "required": ["descriptions"], "additionalProperties": False,
        }}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    cands = json.loads(text)["descriptions"]
    return [c.strip() for c in cands if (not key) or verify(c, key)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True, help="JSON pool cache, shared across splits")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--variants", type=int, default=6)
    args = ap.parse_args()

    pool = {}
    if os.path.exists(args.cache):
        # Keys are repr'd tuples. Use literal_eval, NOT eval: this cache is a shipped,
        # publicly-downloadable artifact, and eval() on it would be arbitrary code execution.
        pool = {ast.literal_eval(k): v for k, v in json.load(open(args.cache)).items()}

    recs = [json.loads(l) for l in open(args.inp)]
    profiles = {}
    for r in recs:
        profiles.setdefault(profile_key(r), []).append(r)
    todo = [k for k in profiles if k not in pool]
    print(f"{len(recs)} clips, {len(profiles)} profiles ({len(todo)} new to paraphrase) via {args.model}")

    # The pool cache ships with the repo (720 profiles). A rerun over the same degradation
    # taxonomy is a pure cache hit, so an API key is only needed when NEW profiles appear.
    client = None
    if todo:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                f"\n  {len(todo)} degradation profiles are not in the cache ({args.cache})\n"
                "  and paraphrasing them needs the Anthropic API.\n\n"
                "    export ANTHROPIC_API_KEY=sk-ant-...\n\n"
                "  (If you did not change the degradation taxonomy or the severity map, the\n"
                "   shipped cache should have covered every profile — this is unexpected.)\n"
            )
        import anthropic
        client = anthropic.Anthropic()
    else:
        print("  all profiles already cached -> no API calls needed")

    for i, key in enumerate(todo):
        try:
            pool[key] = gen_pool(client, args.model, key, args.variants)
        except Exception as e:
            print(f"  profile {i} failed ({type(e).__name__}: {str(e)[:80]}); will use template")
            pool[key] = []
        if (i + 1) % 25 == 0:
            print(f"  paraphrased {i+1}/{len(todo)} profiles")
            json.dump({repr(k): v for k, v in pool.items()}, open(args.cache, "w"))
    json.dump({repr(k): v for k, v in pool.items()}, open(args.cache, "w"))

    n_llm = n_tmpl = 0
    with open(args.out, "w") as f:
        for r in recs:
            key = profile_key(r)
            variants = pool.get(key) or []
            rng = np.random.default_rng(hash(r["id"]) % 2**32)
            if variants:
                desc = variants[int(rng.integers(len(variants)))]
                n_llm += 1
            else:  # grounded template fallback (v2 generator)
                ctx = {"noise_type": (r["params"].get("noise") or {}).get("noise_type")}
                desc = T.describe(r["scores"], rng, ctx)
                n_tmpl += 1
            r["target_text"] = T.build_target(r["scores"], r["mos"], desc)
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} -> {args.out}  (LLM paraphrase {n_llm}, template fallback {n_tmpl})")


if __name__ == "__main__":
    main()
