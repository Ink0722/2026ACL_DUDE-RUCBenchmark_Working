import json
import zai

print(zai.__version__)

'''
content = open("data/data.json", 'r', encoding='utf-8').read().lstrip('\ufeff')
records = json.loads(content)   # 直接得到 list of dict
for rec in records:
    img_field = rec.get("image_path") or rec.get("image") or rec.get("images") or rec.get("img")
    print(rec["type"])
'''