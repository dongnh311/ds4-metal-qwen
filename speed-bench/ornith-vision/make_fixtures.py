#!/usr/bin/env python3
"""Fixtures for the Ornith vision E2E and quality runs (deterministic; /usr/bin/python3 has PIL)."""
import os
import shutil

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")
os.makedirs(OUT, exist_ok=True)
shutil.copyfile(os.path.expanduser("~/models/llama.cpp/tools/mtmd/test-1.jpeg"),
                os.path.join(OUT, "newspaper.jpg"))

CODE = '''def parse_invoice_total(lines):
    total = 0
    for line in lines:
        if line.startswith("INV-2041"):
            total += int(line.split(":")[1])
    return total
'''
mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
img = Image.new("RGB", (900, 260), (30, 30, 30))
ImageDraw.Draw(img).multiline_text((20, 20), CODE, font=mono, fill=(220, 220, 220), spacing=8)
img.save(os.path.join(OUT, "code.png"))

VI = "Hóa đơn số 2041\nNgày 10 tháng 10 năm 2026\nTổng cộng: 1.250.000 đồng"
uni = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 34)
img = Image.new("RGB", (760, 220), (255, 255, 255))
ImageDraw.Draw(img).multiline_text((24, 24), VI, font=uni, fill=(0, 0, 0), spacing=14)
img.save(os.path.join(OUT, "vietnamese.png"))
print("fixtures written to", OUT)
