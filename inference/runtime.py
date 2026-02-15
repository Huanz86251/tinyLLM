from pathlib import Path
import os,json,re,time,hashlib
from project_paths import ROOT,path

os.environ.setdefault('HF_HOME',str(path('cache/huggingface')))
os.environ.setdefault('HF_MODULES_CACHE',str(path('cache/hf_modules')))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
from model.config import Config
from model.model import TinyLLM

def registry():return json.loads(path('configs/models.json').read_text(encoding='utf-8'))
def checkpoint_config(folder,tokenizer):
    raw=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    cfg=Config(**{k:v for k,v in raw.items() if k in Config.__init__.__code__.co_varnames})
    for k,v in raw.items():
        if not hasattr(cfg,k):setattr(cfg,k,v)
    cfg.use_checkpoint=False
    return cfg
def checked_adapter(model,filename,name):
    sd=load_file(str(filename),device='cpu') if filename.suffix=='.safetensors' else torch.load(filename,map_location='cpu',weights_only=True)
    expected={k for k in model.state_dict() if f'.adapters.{name}.' in k}
    if set(sd)!=expected:
        raise RuntimeError(f'Adapter keys mismatch: missing={sorted(expected-set(sd))[:8]}, unexpected={sorted(set(sd)-expected)[:8]}')
    missing,unexpected=model.load_state_dict(sd,strict=False)
    bad=[k for k in missing if f'.adapters.{name}.' in k]
    if bad or unexpected:raise RuntimeError(f'Adapter did not load completely: {bad}, {unexpected}')
    return len(sd)

def load_model(model_id='sft_base',precision='fp32',device=None):
    rec=registry()[model_id]
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if precision=='bf16' and device.type!='cuda':raise ValueError('bf16 demo requires CUDA')
    dtype=torch.float32 if precision=='fp32' else torch.bfloat16
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(str(path(rec['tokenizer'])),local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='left'
    base=path(rec['base'])
    cfg=checkpoint_config(base,tokenizer)
    model=TinyLLM(cfg)
    sd=load_file(str(base/'model.safetensors'),device='cpu')
    # Normalize wrapper names, then require a complete strict base match.
    normalized={k.replace('.base.weight','.weight').replace('.base.bias','.bias'):v for k,v in sd.items()}
    if len(normalized)!=len(sd):raise RuntimeError('Ambiguous duplicate base parameter names')
    sd=normalized
    model.load_state_dict(sd,strict=True)
    del sd
    loaded=0
    if 'adapter' in rec:
        a=rec['adapter']
        model.attach_lora_adapter(adapter_name=a['name'],rank=a['rank'],dropout=a['dropout'],alpha=a['alpha'],target=a['target'])
        loaded=checked_adapter(model,path(a['path']),a['name'])
        model.activate_single_lora(a['name'])
    model.to(device=device,dtype=dtype).eval()
    details={'model_id':model_id,'base':str(base),'adapter':rec.get('adapter'),'adapter_tensors_loaded':loaded,'strict_base_load':True,'precision':precision,'device':str(device),'historical_base_provenance':rec.get('base_provenance','known copied checkpoint')}
    return model,tokenizer,details

def render_prompt(user,system=None):
    s=f'<|im_start|>system\n{system.strip()}<|im_end|>\n' if system else ''
    return s+f'<|im_start|>user\n{user.strip()}<|im_end|>\n<|im_start|>assistant\n'

@torch.inference_mode()
def generate_batch(model,tokenizer,prompts,system=None,max_new_tokens=2048,stop_box=False,vision=None):
    device=next(model.parameters()).device
    ids=[tokenizer.encode(render_prompt(p,system),add_special_tokens=False) for p in prompts]
    width=max(map(len,ids));batch=len(ids)
    inp=torch.full((batch,width),tokenizer.pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(inp)
    for i,row in enumerate(ids):inp[i,-len(row):]=torch.tensor(row,device=device);mask[i,-len(row):]=1
    start=time.perf_counter()
    out=model(input_ids=inp,attention_mask=mask,use_cache=True,past_states=None,**(vision or {}))
    past=out['past_states'];logits=out['logits'][:,-1,:]
    all_ids=[[] for _ in ids];done=[False]*batch
    eos=tokenizer.eos_token_id;im_end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    box=re.compile(r'\\boxed\s*\{\s*[^{}]+?\s*\}')
    for step in range(max_new_tokens):
        logits=logits.clone()
        pad=tokenizer.pad_token_id
        if pad is not None and pad!=eos:logits[:,pad]=-float('inf')
        unk=getattr(model.cfg,'unk_token_id',None)
        if unk is not None and 0<=int(unk)<logits.shape[-1]:logits[:,int(unk)]=-float('inf')
        next_ids=logits.argmax(-1,keepdim=True)
        for i in range(batch):
            if done[i]:next_ids[i]=eos;continue
            tid=int(next_ids[i]);all_ids[i].append(tid)
            text=tokenizer.decode(all_ids[i],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            done[i]=tid in {eos,im_end} or bool(stop_box and box.search(text))
        if all(done) or step+1==max_new_tokens:break
        out=model(input_ids=next_ids,attention_mask=torch.ones_like(next_ids),use_cache=True,past_states=past)
        past=out['past_states'];logits=out['logits'][:,-1,:]
    elapsed=time.perf_counter()-start
    return {'texts':[tokenizer.decode(x,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip() for x in all_ids],'new_tokens':[len(x) for x in all_ids],'seconds':elapsed,'reached_token_limit':[len(x)>=max_new_tokens and not done[i] for i,x in enumerate(all_ids)]}

def log_result(prefix,result):
    folder=path('logs/local_runs');folder.mkdir(parents=True,exist_ok=True)
    filename=folder/(prefix+'_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    filename.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return filename
