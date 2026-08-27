"""Print 'done/total' for one chunk's results.json (works for baseline runs that never set `current`)."""
import json, sys
r, idx, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
N = 150; size = N // n; total = N - idx * size if idx == n - 1 else size
try:
    d = json.load(open(r))
    done = sum(len(p["correct"]) + len(p["wrong"]) for p in d["progress"].values()) + len(set(d.get("skip_indices", [])))
except FileNotFoundError:
    done = 0
print(f"{done}/{total}")
