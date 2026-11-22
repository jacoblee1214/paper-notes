"""논문 부록 표 8-23(실행별 진행도)을 paper/paper.txt에서 읽어 dict로 돌려준다.

paper.txt 는 `pdftotext -layout paper/paper.pdf paper/paper.txt` 로 만든다. 표 값은 논문 v3 기준이다.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIPLET = re.compile(r"(\d+(?:\.\d+)?)/\s*(\d+(?:\.\d+)?)/\s*(\d+(?:\.\d+)?)")
NUM = re.compile(r"(?<![\w/.])(\d+\.\d)(?![\w/.])")


def _blocks():
    lines = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").splitlines()
    # 부록 C: "Table 8: Detailed RQ1 simulation ..." 이후만 본다
    start = next(i for i, l in enumerate(lines) if l.startswith("Table 8: Detailed RQ1 simulation"))
    end = next(i for i, l in enumerate(lines) if l.startswith("D      Taxonomy of Zero-Shot"))
    cur, out = None, {}
    for l in lines[start:end]:
        m = re.match(r"\s*Table (\d+): Detailed", l)
        if m:
            cur = int(m.group(1))
            out[cur] = []
            continue
        if cur is not None:
            out[cur].append(l)
    return out


def parse():
    """반환: {표번호: {행이름: [열별 값]}} ; 8-19는 열별 (run1,run2,run3), 20-23은 열별 float."""
    res = {}
    for n, ls in _blocks().items():
        rows = {}
        for l in ls:
            s = l.strip()
            if not s or s.startswith(("Target embodiment", "Setting", "Each entry", "while columns", "progress", "robots", "pretraining pool", "Rows ")):
                continue
            if 8 <= n <= 19:
                t = TRIPLET.findall(s)
                if not t:
                    continue
                name = TRIPLET.split(s)[0].strip()
                rows[name] = [tuple(float(x) for x in tr) for tr in t]
            else:
                nums = re.findall(r"\d+\.\d", s)
                if not nums:
                    continue
                name = re.split(r"\s+\d+\.\d", s)[0].strip()
                if name.startswith(("Each", "while", "Table")):
                    continue
                rows[name] = [float(x) for x in nums]
        res[n] = rows
    return res


if __name__ == "__main__":
    r = parse()
    for n in sorted(r):
        print(n, len(r[n]), list(r[n])[:9])
