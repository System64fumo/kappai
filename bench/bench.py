#!/usr/bin/env python3
"""Rotation-balanced PP/TG benchmark across trees.

Position within a measurement block dominates PP on this box (whatever runs
first is faster, because the GPU is still at high clocks). So for T trees, block
b runs them in order rotated by b mod T: every tree occupies every position
equally often across blocks. Emits one CSV row per run, tagged with the commit
and the md5 of the engine lib it actually loaded.
"""
import csv, hashlib, os, re, subprocess, sys, time

BASE = os.environ.get("KAPPAI_BENCH_DIR", os.path.expanduser("~/kappai-bench"))
OUT = os.path.join(BASE, "results")
WORDS = ("The quick brown fox jumps over the lazy dog. ") * 280
PROMPT = " ".join(WORDS.split())
RE = re.compile(r"PP:\s*([0-9.]+)\s*t/s.*?TG:\s*([0-9.]+)\s*t/s")


def md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except OSError:
        return "none"


def head(d):
    try:
        return subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "?"


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(d):
    """`.` means the checkout this script lives in; the rest hang off BASE."""
    return REPO if d == "." else os.path.join(BASE, d)


def load_trees():
    trees = []
    with open(os.environ.get("TREES", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "trees.tsv"))) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, d, b, flag, dev, cuda = line.split()
            d = resolve(d)
            trees.append(dict(name=name, dir=d, build=b, flag=flag, dev=dev,
                              cuda=int(cuda), commit=head(d),
                              lib=md5(os.path.join(d, b, "libkappai.so"))))
    return trees


def run(t, model):
    binary = f"{t['dir']}/{t['build']}/kappai-cli"
    if not os.path.exists(binary):
        return None
    cmd = [binary, "-m", model, t["flag"], t["dev"],
           "-c", "4096", "-p", PROMPT, "-n", "8", "-t", "0.0", "--metrics", "pp,tg"]
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(t["dir"], t["build"]))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
    except subprocess.TimeoutExpired:
        return None
    m = RE.search((p.stdout or "") + (p.stderr or ""))
    return m.groups() if m else None


def main():
    model, tag = sys.argv[1], sys.argv[2]
    blocks = int(sys.argv[3])
    cuda_only = len(sys.argv) > 4 and sys.argv[4] == "cuda"
    cpu_only = len(sys.argv) > 4 and sys.argv[4] == "cpu"
    trees = load_trees()
    if cuda_only:
        trees = [t for t in trees if t["cuda"]]
    if cpu_only:
        trees = [t for t in trees if not t["cuda"]]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{tag}.csv"
    rows = []
    present = [t for t in trees if os.path.exists(f"{t['dir']}/{t['build']}/kappai-cli")]
    missing = [t for t in trees if t not in present]
    if missing:
        print("  not built, skipping: " + ", ".join(t["name"] for t in missing))
    print(f"  {tag}: {len(present)} trees x {blocks} blocks, model={os.path.basename(model)}")
    for b in range(blocks):
        order = present[b % len(present):] + present[:b % len(present)]
        for t in order:
            r = run(t, model)
            if not r:
                print(f"    block{b} {t['name']:20s} FAILED (no metrics in output)")
                continue
            pp, tg = r
            print(f"    block{b} {t['name']:20s} PP={pp:>7s} TG={tg:>6s}")
            rows.append(dict(block=b, tree=t["name"], commit=t["commit"], lib=t["lib"],
                             pp=float(pp), tg=float(tg)))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["block", "tree", "commit", "lib", "pp", "tg"])
            w.writeheader()
            w.writerows(rows)
        time.sleep(8)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
