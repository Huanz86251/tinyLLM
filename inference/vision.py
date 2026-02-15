from pathlib import Path
import torch
from PIL import Image,ImageOps
from project_paths import path
from inference.runtime import load_model,generate_batch

def answer_image(filename,prompt,max_new_tokens=2048,precision='bf16',model_id='vlm'):
    filename=Path(filename).expanduser().resolve()
    if not filename.is_file():raise FileNotFoundError(f'Image not found: {filename}')
    # Reuse the original Windows tiling, masks, geometry and InternViT feature code.
    from eval import vllm as legacy
    tower=path('models/vision/InternViT-300M-448px-V2_5')
    for needed in ['config.json','preprocessor_config.json','model.safetensors']:
        if not (tower/needed).is_file():raise FileNotFoundError(tower/needed)
    legacy.VIT_LOCAL_DIR=str(tower)
    legacy.HF_CACHE_DIR=str(path('cache/huggingface'))
    model,tok,info=load_model(model_id,precision)
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    with Image.open(filename) as im0:im=ImageOps.exif_transpose(im0).convert('RGB')
    names,images,meta=legacy.build_views_and_meta(im)
    ex={'vision_meta':meta,'image_files':names}
    vip=legacy.VIPRuntime(str(tower),device,torch.float16 if device.type=='cuda' else torch.float32,True)
    feats=[];masks=[];poses=[];offsets=[]
    for name in names:
        feat=vip.encode_one(images[name]);mask,_=legacy.compute_masks_for_image(ex,name)
        pos,off=legacy.compute_global_pos_for_image(ex,name,mask)
        feats.append(feat);masks.append(torch.from_numpy(mask).bool());poses.append(torch.from_numpy(pos.astype('int64')));offsets.append(torch.from_numpy(off))
    vision={'vision_feats':torch.stack(feats)[None].to(device=device,dtype=dtype),'vision_mask':torch.stack(masks)[None].to(device),'global_pos':torch.stack(poses)[None].to(device),'global_off':torch.stack(offsets)[None].to(device=device,dtype=torch.float16)}
    content='<img>\n'+prompt
    with torch.autocast(device_type=device.type, dtype=dtype if dtype!=torch.float32 else torch.bfloat16, enabled=device.type=='cuda' and dtype!=torch.float32):
        result=generate_batch(model,tok,[content],None,max_new_tokens,vision=vision)
    return {**info,'image':str(filename),'prompt':prompt,'vision_tower':str(tower),'vision_feature_shape':list(vision['vision_feats'].shape),**result}
