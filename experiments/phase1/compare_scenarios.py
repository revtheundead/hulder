"""Compare extraction runs saved by extract_scenario.py. See that file for the commands."""

import torch

D = ".cache/phase1"
POS = {"last": -1, "user_end": -5}  # Qwen template tail: <|im_end|> \n <|im_start|> assistant \n


def load(tag: str) -> dict:
    return torch.load(f"{D}/{tag}.pt", weights_only=False)


def diff(a: list[dict], b: list[dict], pos: int | None) -> tuple[int, float]:
    """(number of prompts not bit-identical, max |a - b|) at one position, or over all positions."""
    n_diff, worst = 0, 0.0
    for ra, rb in zip(a, b):
        assert torch.equal(ra["token_ids"], rb["token_ids"])
        ha, hb = ra["hidden_states"], rb["hidden_states"]
        if pos is not None:
            ha, hb = ha[pos], hb[pos]
        d = (ha.float() - hb.float()).abs().max().item()
        n_diff += d > 0
        worst = max(worst, d)
    return n_diff, worst


def main() -> None:
    off_a, off_b, on_a = load("off_a"), load("off_b"), load("on_a")
    print("layers:", off_a["meta"]["layer_ids"])
    print("template tail (batch, prompt 0):", off_a["meta"]["decoded_tail"][0])

    cases = [
        ("same engine rerun, caching off", off_a["batch"], off_a["batch_rerun"]),
        ("fresh engine, caching off", off_a["batch"], off_b["batch"]),
        ("batch vs single, caching off", off_a["batch"], off_a["single"]),
        ("caching on vs off, first pass", on_a["batch"], off_a["batch"]),
        ("caching on, full-hit rerun vs off", on_a["batch_rerun"], off_a["batch"]),
        ("caching on, single vs off", on_a["single"], off_a["batch"]),
    ]
    n = len(off_a["batch"])
    print(f"\n{'comparison':40s} {'pos':9s} {'prompts differing':>18s} {'max |Δh|':>10s}")
    for name, a, b in cases:
        for pname, p in [*POS.items(), ("all", None)]:
            k, w = diff(a, b, p)
            print(f"{name:40s} {pname:9s} {f'{k}/{n}':>18s} {w:10.4g}")


if __name__ == "__main__":
    main()
