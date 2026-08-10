"""Pick contrasting-style fixture maps and dump their golden Python feature vectors."""
import json, os, shutil, sys
from neural_model import ImprovedBeatmapClassifier

WANT = {
    "stream": {"deathstream", "streams"},
    "jump":   {"cross screen jumps", "large jumps"},
    "tech":   {"slider tech", "tech", "finger control"},
}
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

data = json.load(open("ml_dataset.json", encoding="utf-8"))
clf = ImprovedBeatmapClassifier()

picked = {}
for style, want in WANT.items():
    for s in data:
        bid = s.get("beatmap_id")
        src = os.path.join("downloads", f"downloaded_{bid}.osu")
        if not os.path.exists(src) or bid in picked.values():
            continue
        tags = set(s.get("tags") or [])
        if len(want & tags) >= 2 and 200 < len(s["hit_objects"]) < 2500:
            picked[style] = bid
            shutil.copy(src, os.path.join(out_dir, f"{style}.osu"))
            sections = clf.split_beatmap_into_sections(s["hit_objects"])
            vec = clf._aggregate_features_for_map(sections)
            json.dump(
                {"source": "python", "fixture": style, "beatmap_id": bid,
                 "title": s.get("title"), "tags": sorted(tags),
                 "length": len(vec), "features": [float(x) for x in vec]},
                open(os.path.join(out_dir, f"{style}.python.json"), "w"), indent=2)
            print(f"{style:8} id={bid:10} objs={len(s['hit_objects']):5} {s.get('title')}")
            break
    else:
        print(f"{style:8} NO MATCH")
