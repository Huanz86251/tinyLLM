"""Pinned COIG-CQIA + SmolTalk2 everyday replay; no Fineweb or Tulu.

CPU-only. Preserve previous source filters; save both OPD JSONL and VLM Arrow.
The original replay directory is immutable input, never overwritten.
"""
import sys, json, hashlib, random, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'runtime/minicpm_transformers_449'), str(ROOT)]
from tools.prepare_opd_hybrid_multilingual import audit_chinese, platform_messages, repeated
from tools.prepare_vlm_text_replay import normalize_messages, encode_row


def main():
    import requests
    from transformers import AutoTokenizer
    from datasets import Dataset, load_from_disk
    import pyarrow.parquet as pq
    out=ROOT/'datasets/shared_general_replay_v2'
    if (out/'packed').exists():
        raise RuntimeError('Shared pool already built; use a new version directory before rebuilding')
    out.mkdir(parents=True,exist_ok=True)
    raw=out/'raw'; raw.mkdir(exist_ok=True)
    cfg=json.load(open(ROOT/'configs/opd_hybrid_v3.json',encoding='utf-8'))
    spec=cfg['dataset']['general_sft']; system=cfg['platform_protocol']['system_prompt']
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/text/sft_base_50000'),local_files_only=True,trust_remote_code=True)
    sources=[('zh','m-a-p/COIG-CQIA','8b55868c6168adf86c30e7ca0f782cca1c514297'),
             ('en','HuggingFaceTB/smoltalk2','fc6cc2103c066455aade5d7fbb346039ae36ca5e')]
    selected={'zh':[], 'en':[]}; counts=Counter(); provenance=[]; seen=set()
    for lang,repo,rev in sources:
        response=requests.get(f'https://hf-mirror.com/api/datasets/{repo}/tree/{rev}',params={'recursive':'true','limit':1000},timeout=60)
        response.raise_for_status()
        for entry in response.json():
            name=entry['path']
            eligible=(lang=='zh' and name.endswith('.jsonl') and name.split('/')[0] in ['wiki','zhihu','coig_pc']) or (lang=='en' and 'smoltalk_smollm3_everyday_conversations_no_think-' in name and name.endswith('.parquet'))
            if not eligible: continue
            dest=raw/(lang+'_'+name.replace('/','_'))
            if not dest.exists():
                r=requests.get(f'https://hf-mirror.com/datasets/{repo}/resolve/{rev}/{name}',timeout=(30,180));r.raise_for_status()
                dest.write_bytes(r.content)
            digest=hashlib.sha256(dest.read_bytes()).hexdigest()
            if dest.stat().st_size!=entry['size']: raise RuntimeError('size mismatch '+name)
            expected=entry.get('lfs',{}).get('oid')
            if expected and digest!=expected: raise RuntimeError('hash mismatch '+name)
            provenance.append({'repository':repo,'revision':rev,'file':name,'sha256':digest,'bytes':dest.stat().st_size})
            rows=([json.loads(line) for line in dest.read_text(encoding='utf-8').splitlines() if line.strip()] if lang=='zh' else pq.read_table(dest).to_pylist())
            for i,row in enumerate(rows):
                counts[lang+'_seen']+=1
                if lang=='zh':
                    if re.search(r'分词处理|根据标题写摘要',str(row.get('instruction',''))):
                        counts['zh_drop_segmentation_or_title_only_summary']+=1;continue
                    row['_row_idx']=name+':'+str(i)
                    accepted,reasons=audit_chinese(row,name.split('/')[0],spec,tok,system,seen)
                    counts.update('zh_drop_'+r for r in reasons)
                    if accepted: selected[lang].append(accepted)
                else:
                    messages=normalize_messages(row.get('messages'))
                    if not messages: counts['en_bad_format']+=1;continue
                    prompt='\n'.join(m['content'] for m in messages if m['role']=='user')
                    key=re.sub(r'\s+',' ',prompt).lower()
                    if key in seen: continue
                    if repeated('\n'.join(m['content'] for m in messages if m['role']=='assistant')):continue
                    adapted=[{'role':'system','content':system}]+[{'role':m['role'],'content':('<|thought_start|>\n<|thought_end|>\n' if m['role']=='assistant' else '')+m['content']} for m in messages]
                    n=len(tok.encode(tok.apply_chat_template(adapted,tokenize=False,add_generation_prompt=False),add_special_tokens=False))
                    if n>640:counts['en_too_long']+=1;continue
                    seen.add(key)
                    selected[lang].append({'id':f'everyday:{i}','source':repo,'source_revision':rev,'source_split':name,'messages':adapted,'total_tokens':n,'language':'en'})
            print(name, 'accepted so far',len(selected[lang]),flush=True)
    # Hash based prompt split keeps train/dev disjoint and reproducible.
    pools={}; manifest={'sources':provenance,'files':{},'filters':dict(counts),'excluded':['Fineweb','Tulu'],'seed':20260905}
    packed=[]; schema=load_from_disk(str(ROOT/'datasets/vlm_continual/prepared/packed_vlm_v5')).features
    for lang,full in [('zh','chinese'),('en','english')]:
        random.Random(20260905).shuffle(selected[lang])
        if len(selected[lang])<100:raise RuntimeError('too few '+lang)
        holdout=min(128,max(32,len(selected[lang])//20))
        for split,rows in [('eval',selected[lang][:holdout]),('train',selected[lang][holdout:])]:
            key=full+'_'+split; p=out/(key+'.jsonl')
            p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            manifest['files'][key]={'path':str(p),'rows':len(rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for r in rows:
                # VLM uses ordinary ChatML. OPD keeps its existing empty thought protocol.
                messages=[{'role':m['role'],'content':m['content'].replace('<|thought_start|>\n<|thought_end|>\n','')} for m in r['messages'] if m['role']!='system']
                encoded,reason=encode_row(tok,messages,640,1)
                if encoded is None:raise RuntimeError(reason)
                ids,mask=encoded
                packed.append({'index':10000000+len(packed),'image_file':[],'dataset':'shared_replay_'+lang,'lang':lang,'has_cot':False,'split':split,'vision_meta':'{}','input_ids':ids,'input_len':len(ids),'target_mask':mask})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    target=out/'packed'
    if target.exists():raise RuntimeError('packed already exists; do not overwrite')
    Dataset.from_list(packed,features=schema).save_to_disk(str(target))
    print(json.dumps({'files':manifest['files'],'filters':dict(counts)},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
