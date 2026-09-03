import re

with open('/home/ruben/Desktop/unibotics/docs/assets/articles/2026_article_attempt/MTAP/CAV_manuscript.bbl', 'r') as f:
    content = f.read()

bibitems = re.findall(r'\\bibitem(?:\[.*?\])?\{([a-zA-Z0-9_-]+)\}', content)
print(f"Total bibitems in .bbl: {len(bibitems)}")
for idx, key in enumerate(bibitems, 1):
    print(f"{idx}: {key}")
