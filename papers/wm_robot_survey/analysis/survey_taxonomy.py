"""Survey 03(a): reference parsing, per-section citation counts, year/venue distribution.

Input : paper/paper.txt (pdftotext -layout of arXiv:2605.00080v1)
Output: analysis/survey_taxonomy_results.json (printed summary too)
Run   : python3 -I analysis/survey_taxonomy.py   (from the folder root)
"""
import json, re, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
lines = open(os.path.join(ROOT, "paper", "paper.txt"), encoding="utf-8").read().split("\n")

ref_start = next(i for i, l in enumerate(lines) if l.strip() == "References")
body = lines[:ref_start]
refs_raw = lines[ref_start + 1:]

# ---- 1. split references into entries (entry starts at indent 0, capitalised, not a footer number)
entries = []
cur = None
for l in refs_raw:
    if not l.strip():
        continue
    if re.match(r"^\s{20,}\d+\s*$", l):   # page number footer
        continue
    if re.match(r"^[A-ZÀ-ɏ][^\s]", l) and (len(l) - len(l.lstrip()) == 0):
        if cur:
            entries.append(cur)
        cur = l.strip()
    elif cur is not None:
        cur += " " + l.strip()
if cur:
    entries.append(cur)

def first_author(e):
    e = re.sub(r"^arXiv:\S+\s+(?=(?:19|20)\d{2}\s)", "", e)  # stray arXiv id before entry
    e = re.sub(r"^(?:(?:19|20)\d{2}\s+)", "", e)
    m = re.match(r"^(.*?)(,| and | et al\.)", e)
    name = m.group(1) if m else e.split()[0]
    return name.split()[-1].strip(".")

def year_of(e):
    # year = first 19xx/20xx token followed by '.' or ',' or ')' that appears after the author list
    cands = re.findall(r"(?<![\w.])((?:19|20)\d{2})[a-z]?(?=[.,)\s])", e)
    return int(cands[0]) if cands else None

VENUE_KEYS = [
    ("arXiv", r"arXiv preprint|arXiv:|\barxiv\b"),
    ("CVPR", r"CVPR|Computer Vision and Pattern Recognition"),
    ("ICCV", r"ICCV|International Conference on Computer Vision"),
    ("ECCV", r"ECCV|European Conference on Computer Vision"),
    ("NeurIPS", r"NeurIPS|Neural Information Processing|NIPS"),
    ("ICLR", r"ICLR|International Conference on Learning Representations"),
    ("ICML", r"ICML|International Conference on Machine Learning"),
    ("CoRL", r"CoRL|Conference on Robot Learning"),
    ("RSS", r"Robotics: Science and Systems|RSS\b"),
    ("ICRA", r"ICRA|International Conference on Robotics and Automation"),
    ("IROS", r"IROS|Intelligent Robots and Systems"),
    ("RA-L", r"Robotics and Automation Letters|RA-L"),
    ("AAAI", r"AAAI"),
    ("journal/other", r"Science|Nature|Transactions|Journal|IJRR|Machine Intelligence"),
]
def venue_of(e):
    for name, pat in VENUE_KEYS:
        if re.search(pat, e):
            return name
    return "other"

R = []
for i, e in enumerate(entries):
    m_ax = re.search(r"arXiv:\s?(\d{4}\.\d{4,5})", e)
    R.append({"idx": i, "first": first_author(e), "year": year_of(e), "venue": venue_of(e), "text": e[:160], "arxiv": m_ax.group(1) if m_ax else None})

# ---- 2. citations in body: "Surname et al., 2025a, 2026" / "Surname and Surname, 2018" / "Surname, 2023"
sections = [
    ("1 Introduction", 1, 0), ("2 Background", 2, 0), ("3 World Model for Policy", 3, 0),
    ("4 World Model as Simulator", 4, 0), ("5 Robotic Video Generation", 5, 0),
    ("6 Other Applications", 6, 0), ("7 Benchmarks, Datasets, Results", 7, 0), ("8 Challenges", 8, 0),
]
# locate heading line numbers
heads = {}
for i, l in enumerate(body):
    m = re.match(r"^\s*([1-8])\s{2,}([A-Z][A-Za-z ,&\-]+)$", l)
    if m and int(m.group(1)) not in heads and i > 60:
        heads[int(m.group(1))] = i
order = sorted(heads.items())
def section_of(line_no):
    sec = 1  # introduction precedes the first numbered heading found
    if heads.get(2) is not None and line_no < heads[2]:
        return 1
    for num, ln in order:
        if line_no >= ln:
            sec = num
    return sec

cite_re = re.compile(r"([A-Z][A-Za-z\-À-ɏ']+)(?: et al\.| and [A-Z][A-Za-z\-À-ɏ']+)?,? \(?((?:19|20)\d{2}[a-z]?(?:, (?:19|20)\d{2}[a-z]?)*)\)?")
cites = collections.defaultdict(set)   # section -> set of (surname, year)
all_cites = set()
# join lines so citations split across line breaks are found; map offsets back to line numbers
offsets, pos = [], 0
for l in body:
    offsets.append(pos)
    pos += len(l) + 1
text = "\n".join(body)
for m in cite_re.finditer(text):
    line_no = max(i for i, o in enumerate(offsets) if o <= m.start())
    sec = section_of(line_no)
    surname = m.group(1)
    years = [int(re.match(r"(\d{4})", y.strip()).group(1)) for y in m.group(2).split(",")]
    for y in years:
        cites[sec].add((surname, y))
        all_cites.add((surname, y))

unmatched = collections.Counter()

def match_ref(surname, year):
    hits = [r for r in R if r["first"] == surname and r["year"] == year]
    return hits

matched = {}
missing = []
for key in sorted(all_cites):
    h = match_ref(*key)
    if len(h) == 1:
        matched[key] = h[0]["idx"]
    elif len(h) == 0:
        missing.append(key)
    else:
        matched[key] = h[0]["idx"]  # ambiguous: keep first, record
        unmatched[f"ambiguous {key}"] += 1

cited_refs = set(matched.values())
per_section = {}
for sec, keys in sorted(cites.items()):
    refs_in = {matched[k] for k in keys if k in matched}
    per_section[sec] = {
        "name": dict(sections)[0] if False else None,
        "unique_citation_keys": len(keys),
        "unique_refs_matched": len(refs_in),
    }
names = {1: "1 Introduction", 2: "2 Background", 3: "3 World Model for Policy", 4: "4 World Model as Simulator",
         5: "5 Robotic Video Generation", 6: "6 Other Applications", 7: "7 Benchmarks/Datasets/Results", 8: "8 Challenges"}
for s in per_section:
    per_section[s]["name"] = names.get(s, str(s))

years = collections.Counter(r["year"] for r in R)
venues = collections.Counter(r["venue"] for r in R)
ven_cited = collections.Counter(R[i]["venue"] for i in cited_refs)
yr_cited = collections.Counter(R[i]["year"] for i in cited_refs)


# ---- 3. subsection-level coverage (3.1, 5.2, ...) and per-subsection year/venue stats
subheads = []
for i, l in enumerate(body):
    m = re.match(r"^\s*(\d\.\d(?:\.\d)?)\s{2,}([A-Z][^\n]{3,80})$", l)
    if m and i > 60 and int(m.group(1)[0]) >= 2:
        subheads.append((m.group(1), i, m.group(2).strip()))
def sub_of(line_no):
    cur = None
    for num, ln, name in subheads:
        if line_no >= ln:
            cur = (num, name)
    return cur
sub_keys = collections.defaultdict(set)
for m in cite_re.finditer(text):
    line_no = max(i for i, o in enumerate(offsets) if o <= m.start())
    sb = sub_of(line_no)
    if sb is None:
        continue
    for y in [int(re.match(r"(\d{4})", yy.strip()).group(1)) for yy in m.group(2).split(",")]:
        sub_keys[sb].add((m.group(1), y))
per_sub = {}
for (num, name), keys in sorted(sub_keys.items(), key=lambda kv: [int(x) for x in kv[0][0].split(".")]):
    idxs = {matched[k] for k in keys if k in matched}
    per_sub[num] = {
        "title": name,
        "unique_citation_keys": len(keys),
        "unique_refs_matched": len(idxs),
        "venue_counts": dict(collections.Counter(R[i]["venue"] for i in idxs).most_common()),
        "year_min": min((R[i]["year"] for i in idxs if R[i]["year"]), default=None),
        "year_median": sorted(R[i]["year"] for i in idxs if R[i]["year"])[len(idxs)//2] if idxs else None,
        "share_2025_2026": round(sum(1 for i in idxs if R[i]["year"] in (2025, 2026)) / max(1, len(idxs)), 3),
        "ref_idx": sorted(idxs),
    }

out = {
    "per_subsection": per_sub,
    "reference_entries_parsed": len(R),
    "reference_entries_with_year": sum(1 for r in R if r["year"]),
    "unique_in_text_citation_keys": len(all_cites),
    "matched_to_reference": len(matched),
    "unmatched_citation_keys": sorted(f"{a} {y}" for a, y in missing),
    "ambiguous_or_multi": dict(unmatched),
    "reference_entries_cited_in_body": len(cited_refs),
    "reference_entries_not_cited_in_body_text": len(R) - len(cited_refs),
    "per_section": per_section,
    "year_distribution_all_refs": dict(sorted(years.items(), key=lambda x: (x[0] is None, x[0] or 0))),
    "venue_distribution_all_refs": dict(venues.most_common()),
    "venue_distribution_cited_refs": dict(ven_cited.most_common()),
    "year_distribution_cited_refs": dict(sorted(yr_cited.items(), key=lambda x: (x[0] is None, x[0] or 0))),
    "section_heading_lines": {str(k): v for k, v in heads.items()},
    "section_cited_ref_idx": {str(k): sorted(matched[x] for x in v if x in matched) for k, v in cites.items()},
    "section_unresolved_keys": {str(k): sorted(f"{a} {y}" for a, y in v if (a, y) not in matched) for k, v in cites.items()},
}
json.dump(out, open(os.path.join(HERE, "survey_taxonomy_results.json"), "w"), ensure_ascii=False, indent=1)
json.dump([{"idx": r["idx"], "first": r["first"], "year": r["year"], "venue": r["venue"], "arxiv": r["arxiv"], "text": r["text"]} for r in R],
          open(os.path.join(HERE, "survey_refs_parsed.json"), "w"), ensure_ascii=False, indent=0)
print(json.dumps({k: v for k, v in out.items() if k not in ("unmatched_citation_keys",)}, ensure_ascii=False, indent=1))
print("unmatched:", len(missing), out["unmatched_citation_keys"][:60])
