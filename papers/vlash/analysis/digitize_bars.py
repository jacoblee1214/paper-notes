"""논문 그림(막대 그래프)의 래스터 이미지에서 막대 높이와 오차 막대를 읽는다.

사용: pdfimages -png -f 5 -l 6 paper.pdf out/f 로 이미지를 뽑은 뒤
  python3 digitize_bars.py out/f-002.png 354.5 1205.5 320   (그림 7: 100% 행, 0% 행, 범례 아래 시작 행)
  python3 digitize_bars.py out/f-004.png 245.5 1094.25 215  (그림 8)
출력: (방법, 막대 중심 x, 막대 높이 %, 오차 막대 위 행, 아래 행). 눈금 행은 격자선(231,231,231)에서 읽었다.
"""
import sys, json
from PIL import Image
import numpy as np
fn,y100,y0=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
ytop=int(sys.argv[4]) if len(sys.argv)>4 else 320
a=np.array(Image.open(fn).convert('RGB')).astype(int)
cols={'VLASH':(143,195,122),'RTC':(178,216,157),'Sync':(228,153,152),'Naive':(237,190,115)}
H,W,_=a.shape
res=[]
blk=(a.sum(2)<40)
for n,c in cols.items():
    m=(np.abs(a-np.array(c)).sum(2)<4); m[:ytop]=False
    xs=np.where(m.sum(0)>3)[0]
    groups=[];s=None;p=None
    for x in xs:
        if s is None: s=x
        elif x!=p+1: groups.append((s,p)); s=x
        p=x
    if s is not None: groups.append((s,p))
    for x0,x1 in groups:
        if x1-x0<25: continue
        cx=(x0+x1)//2
        rows=np.where(m[:,x0:x1+1].sum(1)>10)[0]
        top=rows.min()-1.5
        # error bar: black in center +-2 cols above bar
        sub=blk[ytop:,cx-3:cx+4].sum(1)>=3
        er=np.where(sub)[0]+ytop
        # error bar extents: contiguous black center line
        # find cap rows: rows with wide black (>=15 wide) near center
        capm=blk[:,cx-15:cx+16].sum(1)>=25
        capm[:ytop]=False
        caps=np.where(capm)[0]
        caps=[r for r in caps if r<y0-4]
        v=(y0-top)/(y0-y100)*100
        res.append((n,cx,round(v,1),caps[:1],caps[-1:]))
res.sort(key=lambda r:r[1])
for r in res: print(r)
