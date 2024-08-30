import json

def JSON2Bboxes(json_path):
    with open(json_path,"r") as json_fin:
        json_dict=json.load(json_fin)

    if not isinstance(json_dict, dict) or "bboxes" not in json_dict:
        raise ValueError(f"Invalid JSON format or 'bboxes' key not found in {json_path}")
    
    return json_dict["bboxes"]

def Bboxes2JSON(bboxes,json_path):
    try:bboxes=bboxes.tolist()
    except:pass
    # for box in bboxes:
    #     for i in range(5):
    #         box[i] = int(round(float(box[i]), 0))
    json_dict={}
    json_dict["bboxes"]=bboxes
    with open(json_path,"w") as fout:
        json.dump(json_dict,fout,ensure_ascii=False) 
    return 