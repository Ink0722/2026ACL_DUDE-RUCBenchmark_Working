# ...existing code...
import os
import copy
import json
import time
from PIL import Image
from datasets import Dataset
from .rule import generate_clicks 
from .rule import generate_clicks_2
from .rule import generate_empty_clicks


# 这里Train参数主要影响的是后续要不要生成空样本，事实上对于我们的情况来说不管怎么样都得生成，所以这里的参数值直接改成True
def load_local_dataset(ann_path="data/Deception.json", images_dir="data/images", load_images=True, Train=True):
    """
    load_images: 是否把图片读取为 PIL.Image 对象并放入 records["images"]
    """
    ann_dir = os.path.dirname(os.path.abspath(ann_path))
    content = open(ann_path, 'r', encoding='utf-8').read().lstrip('\ufeff')
    initrecords = json.loads(content)   # 直接得到 list of dict
    records = []

    for rec in initrecords:

        # 候选框为空的情况要略去，这个后面要改
        if not rec.get("correct_box") or not rec["correct_box"].get("bbox"): continue
        if rec.get("id") == 998: continue
        
        img_field = rec.get("image_path") or rec.get("image") or rec.get("images") or rec.get("img")
        if img_field is None:
            rec["images"] = []
            rec["image_path_normalized"] = []
        else:
            if isinstance(img_field, str):
                img_list = [img_field]
            else:
                img_list = list(img_field)

            norm_paths = []
            images_objs = []
            for p in img_list:
                if p is None:
                    continue
                p = p.replace("\\", "/")
                if p.startswith("./"):
                    p = p[2:]

                # 尝试若干候选路径
                candidates = [
                    os.path.normpath(os.path.join(ann_dir, p)),
                    os.path.normpath(os.path.join(os.getcwd(), p)),
                    os.path.normpath(p),
                    os.path.normpath(os.path.join(images_dir, p)),
                    os.path.normpath(os.path.join(images_dir, os.path.basename(p))),
                ]

                found = None
                for c in candidates:
                    if os.path.exists(c):
                        found = c
                        break

                if found is None:
                    # 保持最后一个候选路径以便 later debugging
                    found = candidates[-1]

                norm_paths.append(found)

                if load_images:
                    try:
                        img = Image.open(found).convert("RGB")
                        images_objs.append(img)
                        # print("🌳Successful load images!")
                    except Exception:
                        # 若无法打开，记录 None（或可记录占位图）
                        images_objs.append(None)
                        print("❌Not Successful load images!")

            rec["images"] = images_objs if load_images else norm_paths
            rec["image_path_normalized"] = norm_paths
        
        if Train and rec.get("dark_box") and rec["dark_box"].get("bbox"):
            # correct and dark box all exists
            rec_d = copy.deepcopy(rec)

            clicks = generate_clicks(rec)
            b_click,d_click = clicks["benign"], clicks["deceptive"]
            n_clicks = generate_empty_clicks(rec)
            
            rec["click"] = b_click["coordinates"]
            rec["gen_type"] = 1
            
            rec_d["click"] = d_click["coordinates"]
            rec_d["gen_type"] = -1

            records.append(rec)
            records.append(rec_d)

            for click in n_clicks:
                rec_n = copy.deepcopy(rec)
                rec_n["click"] = click["coordinates"]
                rec_n["gen_type"] = 0
                records.append(rec_n)

        elif Train:
            # only correct box exists
            
            clicks = generate_clicks_2(rec)
            b_click = clicks["benign"]
            n_clicks = generate_empty_clicks(rec)
            
            rec["click"] = b_click["coordinates"]
            rec["gen_type"] = 1

            records.append(rec)

            for click in n_clicks:
                rec_n = copy.deepcopy(rec)
                rec_n["click"] = click["coordinates"]
                rec_n["gen_type"] = 0
                records.append(rec_n)
        else:
            records.append(rec)
    
    print(records[:1])

    if Train:
        for i,d in enumerate(records):
            new_sp = (
                "You are an experience optimizer for a web browsing agent."
                "Given the screenshot, the user task and output click coordinates, "
                "Your job is to identify if the click is right to finish user's task(1) , unable to finish user's task(0), or suspicious for potential deceptions(-1)."
                "You should also provide the confident score(0-1) of your judgement."
                "You can think in less than 50 words and need to output exactly the format like:"
                "<judge>1</judge>, <conf>0.35</conf>"
                "or like"
                "<judge>0</judge>, <conf>0.45</conf>"
                "<judge>-1</judge>, <conf>0.63</conf>"
                "<judge>1</judge>, <conf>0.87</conf>...."
                "Note that you should not be overly confident in your output."
                "The confidence score of your output should generally be between 0.3-0.8."
                "Of course, if you are particularly confident or unsure, you can exceed this threshold."
                "DO NOT FORGET THE XML MARK <judge></judge> or <conf></conf> when output!!"
                "Now here is your inputs:"
            )
            records[i]["messages"][0]["content"] = new_sp
            # records[i]["messages"][1]["content"] = f'''Previous experience: {" "}. Output click: {d["click"]}. User task: ''' + records[i]["messages"][1]["content"]
            records[i]["messages"][1]["content"] = f'''Output click: {d["click"]}. User task: ''' + records[i]["messages"][1]["content"]

    ds = Dataset.from_list(records)
    print("[DST] Show 1 sample",ds[0])

    return ds

def split_batch(samples, batch_size):
    single_elements = []
    for i in range(batch_size):
        element = {key: samples[key][i] for key in samples.keys()}
        single_elements.append(element)
    return single_elements