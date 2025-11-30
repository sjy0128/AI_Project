import os
import struct
from PIL import Image, ImageDraw
from tqdm import tqdm
import os

base = "./raw_data/Gnt1.0Test"
out_base = "./data/test"
os.makedirs(out_base, exist_ok=True)

def gb2312_to_unicode_hex(code):
    return "0x" + format(code, '04X')

def render_sample(strokes, size=64, margin=4):
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    if not xs: return img

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    w = maxx - minx + 1
    h = maxy - miny + 1
    scale = (size - margin*2) / max(w, h)

    for s in strokes:
        if len(s) > 1:
            p = [( (p[0]-minx)*scale+margin, (p[1]-miny)*scale+margin ) for p in s]
            draw.line(p, fill=0, width=2)
    return img

def read_gnt(fp):
    while True:
        header = fp.read(10)
        if not header:
            break
        sample_size = struct.unpack("<I", header[0:4])[0]
        tagcode = struct.unpack(">H", header[4:6])[0]
        width = struct.unpack("<H", header[6:8])[0]
        height = struct.unpack("<H", header[8:10])[0]
        bitmap = fp.read(sample_size-10)
        arr = list(bitmap)
        img = Image.new("L", (width, height))
        img.putdata([(255-x) for x in arr])
        yield tagcode, img

files = sorted([f for f in os.listdir(base) if f.endswith("-t.gnt")])

cnt = 0
for fname in tqdm(files):
    path = os.path.join(base, fname)
    with open(path, "rb") as f:
        for tagcode, img in read_gnt(f):
            label = gb2312_to_unicode_hex(tagcode)
            outdir = os.path.join(out_base, label)
            os.makedirs(outdir, exist_ok=True)

            img = img.resize((64,64), Image.BILINEAR)

            outpath = os.path.join(outdir, f"{cnt:07d}.png")
            img.save(outpath)
            cnt+=1

print("DONE")
