"""
tag_fingerprint.py -- empirically characterize each tag from the labeled dataset.

For every tag: split maps into has-tag vs not, compute Cohen's d per feature on the
max-pooled 29-feature block, and report the separating features + example maps + a
"blindness" score (best |d| any feature achieves). Low blindness => current features
cannot see the tag => that's what we need to add.
"""
import json
import numpy as np
from collections import defaultdict
from neural_model import ImprovedBeatmapClassifier

FEAT = [
    "burst_count","stream_count","max_cont_stream","total_stream_notes",
    "rhythm_change_ratio","global_rhythm_var","max_stream_spc_var","buzz_slider_count",
    "finger_control","avg_rhythm_instab","avg_spacing_instab","slider_disrupt_rate",
    "num_objects","objects_per_sec","mean_distance","std_distance","p95_distance",
    "mean_time_gap","std_time_gap","slider_ratio","mean_angle","std_angle",
    "sharp_angle_r","square_angle_r","wide_angle_r","linear_angle_r",
    "vertical_jump_r","perfect_overlap_r","true_linear_r",
]
EXCLUDE = {"comfortable", "practise"}

clf = ImprovedBeatmapClassifier()
data = json.load(open("ml_dataset.json", encoding="utf-8"))

X, tags_per_map, titles = [], [], []
for s in data:
    t = [x for x in (s.get("tags") or []) if x not in EXCLUDE]
    if not t:
        continue
    sections = clf.split_beatmap_into_sections(s["hit_objects"])
    if not sections:
        continue
    vec = clf._aggregate_features_for_map(sections)
    if vec is None:
        continue
    X.append(np.asarray(vec, dtype=float)[:29])   # max block only
    tags_per_map.append(set(t))
    titles.append(s.get("title", s.get("beatmap_id", "?")))

X = np.vstack(X)
N = len(X)
print(f"maps analyzed: {N}   features: {X.shape[1]} (max-pooled)\n")

tag_counts = defaultdict(int)
for ts in tags_per_map:
    for t in ts:
        tag_counts[t] += 1

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled

rows = []
for tag, cnt in tag_counts.items():
    mask = np.array([tag in ts for ts in tags_per_map])
    withg, without = X[mask], X[~mask]
    ds = np.array([cohens_d(withg[:, j], without[:, j]) for j in range(29)])
    order = np.argsort(-np.abs(ds))
    best = abs(ds[order[0]])
    top = [(FEAT[j], ds[j]) for j in order[:3]]
    ex = [titles[i] for i in range(N) if mask[i]][:5]
    rows.append((best, tag, cnt, top, ex))

rows.sort()  # ascending best|d| => blindest tags first

print("=" * 100)
print("TAGS RANKED BY BLINDNESS (weakest separating feature first)")
print("=" * 100)
print(f"{'tag':<24}{'n':>5}  {'best|d|':>7}  top separating features (d = with-minus-without, in SDs)")
print("-" * 100)
for best, tag, cnt, top, ex in rows:
    tops = ", ".join(f"{n}({d:+.2f})" for n, d in top)
    print(f"{tag:<24}{cnt:>5}  {best:>7.2f}  {tops}")

print("\n\n" + "=" * 100)
print("EXAMPLE MAPS PER TAG (verify my understanding against these)")
print("=" * 100)
for best, tag, cnt, top, ex in rows:
    print(f"\n[{tag}]  n={cnt}  best|d|={best:.2f}")
    print("   top feat: " + ", ".join(f"{n}({d:+.2f})" for n, d in top))
    for e in ex:
        print(f"     - {e}")
