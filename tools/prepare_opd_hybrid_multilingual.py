"""Prepare the reviewed 2:1 Chinese/English replay pool for OPD hybrid v3."""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project_paths import path

THOUGHT_START='<|thought_start|>'
THOUGHT_END='<|thought_end|>'
FORBIDDEN=('<|im_start|>','<|im_end|>',THOUGHT_START,THOUGHT_END)
MATH_RE=re.compile(r'(?:\\boxed|\b(?:solve|calculate|equation|algebra|geometry|derivative|integral|proof)\b|(?:求解|计算|方程|几何|导数|积分|证明|概率|因式分解)|\d\s*[+*/=]\s*\d)',re.I)
HIGH_RISK_RE=re.compile(r'(?:诊断|治疗|药物|用药|疾病|症状|医院|医生|法律|律师|诉讼|判刑|合同纠纷|股票|基金|证券|投资建议|贷款|保险|收益率|政治|选举|总统|主席|政府政策|军事|战争|自杀|自残|杀人|伤害他人|炸弹|爆炸物|毒品|制毒|歧视|仇恨|色情|强奸|犯罪|违法|武器)')
CURRENT_RE=re.compile(r'(?:最新|目前|现任|今年|今日|实时|价格|票价|汇率|截止.{0,4}年|20(?:2[4-9]|3\d)年)')
NOISE_RE=re.compile(r'(?:关注.{0,8}(?:账号|博主)|点赞.{0,8}收藏|抽奖|优惠券|下单|购买链接|扫码|加微信|作为一个AI|语言模型|无法回答)')
SOURCE_DEP_RE=re.compile(r'(?:根据上文|根据本文|根据所给材料|如图所示|见下图|本书作者|这篇文章中)')


def read_json(file): return json.loads(Path(file).read_text(encoding='utf-8-sig'))
def read_jsonl(file):
    with Path(file).open('r',encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def write_jsonl(file,rows):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('w',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
def sha256(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def normalized(text): return re.sub(r'[ \t]+',' ',str(text or '').replace('\r\n','\n').replace('\r','\n')).strip()
def cjk_ratio(text):
    cjk=len(re.findall(r'[\u3400-\u9fff]',text));letters=len(re.findall(r'[A-Za-z\u3400-\u9fff]',text))
    return cjk/max(letters,1)
def repeated(text):
    compact=re.sub(r'\s+',' ',text)
    if re.search(r'(?P<x>.{20,140}?)(?:\s*\1){2,}',compact,re.S):return True
    lines=[x.strip() for x in text.splitlines() if len(x.strip())>=12]
    return len(lines)!=len(set(lines))
def platform_messages(user,answer,system):
    return [{'role':'system','content':system},{'role':'user','content':user},
            {'role':'assistant','content':THOUGHT_START+'\n'+THOUGHT_END+'\n'+answer}]


def request_json(session,url,params=None):
    last=None
    for attempt in range(5):
        try:
            response=session.get(url,params=params,timeout=60);response.raise_for_status();return response.json()
        except Exception as exc:
            last=exc;time.sleep(2*(attempt+1))
    raise RuntimeError(f'network request failed: {last}')


def download_chinese(cfg,output):
    import requests
    zh=cfg['chinese'];session=requests.Session()
    meta=request_json(session,'https://huggingface.co/api/datasets/'+zh['repository'])
    if meta.get('sha')!=zh['revision']:raise RuntimeError('COIG-CQIA revision changed')
    files={}
    for component in zh['components']:
        probe=request_json(session,'https://datasets-server.huggingface.co/rows',{
            'dataset':zh['repository'],'config':component['config'],'split':'train','offset':0,'length':1})
        total=int(probe.get('num_rows_total',0));starts=list(range(0,total,100))
        random.Random(cfg['seed']+sum(map(ord,component['config']))).shuffle(starts)
        collected=[]
        for start in starts:
            batch=request_json(session,'https://datasets-server.huggingface.co/rows',{
                'dataset':zh['repository'],'config':component['config'],'split':'train',
                'offset':start,'length':min(100,total-start)}).get('rows',[])
            for item in batch:
                row=item.get('row',{});row['_row_idx']=item.get('row_idx');collected.append(row)
            if len(collected)>=component['raw_rows']:break
        collected=collected[:component['raw_rows']]
        file=output/f"zh_{component['config']}_raw.jsonl";write_jsonl(file,collected)
        files[component['config']]={'path':str(file),'rows':len(collected),'total_source_rows':total,'sha256':sha256(file)}
    manifest={'repository':zh['repository'],'revision':zh['revision'],'transport':'datasets-server rows API',
              'bounded_download':True,'files':files}
    (output/'zh_raw_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'downloaded_chinese',**manifest},ensure_ascii=False,indent=2))


def audit_chinese(row,component,cfg,tok,system,seen):
    reasons=[]
    if cfg['chinese']['require_human_verified'] and component!='coig_pc' and row.get('human_verified') is not True:reasons.append('not_human_verified')
    if cfg['chinese']['require_human_answer'] and str(row.get('answer_from','')).lower() not in ('human','expert'):reasons.append('not_human_answer')
    user=normalized(row.get('instruction'))
    extra=normalized(row.get('input'))
    if extra:user=user+'\n'+extra
    answer=normalized(row.get('output'))
    if not user or not answer:reasons.append('empty')
    joined=user+'\n'+answer
    if any(x in joined for x in FORBIDDEN):reasons.append('protocol_injection')
    if '\ufffd' in joined or '\x00' in joined:reasons.append('encoding_noise')
    if cjk_ratio(joined)<0.65:reasons.append('not_chinese_dominant')
    if MATH_RE.search(user):reasons.append('math_like')
    if HIGH_RISK_RE.search(joined):reasons.append('high_risk')
    if CURRENT_RE.search(joined):reasons.append('time_sensitive')
    if NOISE_RE.search(joined):reasons.append('ad_or_boilerplate')
    if SOURCE_DEP_RE.search(joined):reasons.append('missing_context')
    if '```' in joined or '<html' in joined.lower() or 'http://' in joined or 'https://' in joined:reasons.append('code_or_link')
    if repeated(answer):reasons.append('repetition')
    key=re.sub(r'\s+',' ',user).lower()
    if key in seen:reasons.append('duplicate_prompt')
    adapted=platform_messages(user,answer,system)
    rendered=tok.apply_chat_template(adapted,tokenize=False,add_generation_prompt=False)
    total_tokens=len(tok.encode(rendered,add_special_tokens=False))
    user_tokens=len(tok.encode(user,add_special_tokens=False));answer_tokens=len(tok.encode(answer,add_special_tokens=False))
    if total_tokens>cfg['max_total_tokens'] or user_tokens>256 or answer_tokens>cfg['max_assistant_tokens']:reasons.append('too_long')
    if answer_tokens<cfg['min_assistant_tokens']:reasons.append('too_short')
    if reasons:return None,reasons
    seen.add(key)
    checks=[('source_subset_human_quality_verified' if component=='coig_pc' else 'source_row_human_verified'),'answer_origin_recorded','chinese_dominant','not_math','not_high_risk',
            'not_time_sensitive','not_advertising','self_contained','no_repeat','length_ok','protocol_safe']
    return {'id':f"coig:{component}:{row.get('_row_idx')}",'language':'zh','source':cfg['chinese']['repository'],
            'source_revision':cfg['chinese']['revision'],'source_config':component,
            'license':cfg['chinese']['license'],'messages':adapted,'total_tokens':total_tokens,
            'assistant_tokens':[answer_tokens],'reasoning_policy':'empty_boundary_original_answer',
            'review':{'status':'accepted','basis':'COIG human verification plus exhaustive local row audit','checks':checks}},[]


def build_english(cfg,system):
    old=path('datasets/opd_hybrid_v3');rows=[];seen=set()
    high_risk=re.compile(r'\b(?:diagnos|treat(?:ment)?|medication|disease|symptom|doctor|hospital|legal advice|lawsuit|contract dispute|stock|fund|securities|investment advice|loan|insurance|election|president|government policy|military|war)\w*\b',re.I)
    current=re.compile(r'\b(?:latest|currently|today|this year|real-time|current price|recent games?|as of 20(?:2[4-9]|3\d))\b',re.I)
    noise=re.compile(r'\b(?:follow my|like and subscribe|discount code|buy now|giveaway|as an ai|language model|cannot answer)\b',re.I)
    for component in cfg['english']['components']:
        name=component['name'];need=component['rows'];candidates=[]
        source=[]
        for split in ('train','eval'):source.extend(read_jsonl(old/f'{name}_{split}.jsonl'))
        random.Random(cfg['seed']+71+sum(map(ord,name))).shuffle(source)
        for original in source:
            row=dict(original);messages=row['messages'];joined='\n'.join(m['content'] for m in messages)
            users='\n'.join(m['content'] for m in messages if m['role']=='user')
            answers='\n'.join(m['content'].replace(THOUGHT_START,'').replace(THOUGHT_END,'') for m in messages if m['role']=='assistant')
            key=re.sub(r'\s+',' ',users).lower();bad=[]
            if cjk_ratio(joined)>0.10:bad.append('not_english_dominant')
            if MATH_RE.search(users):bad.append('math_like')
            if high_risk.search(joined):bad.append('high_risk')
            if current.search(users):bad.append('time_sensitive')
            if noise.search(joined):bad.append('ad_or_boilerplate')
            if SOURCE_DEP_RE.search(joined):bad.append('missing_context')
            if repeated(answers):bad.append('repetition')
            if key in seen:bad.append('duplicate_prompt')
            if row['total_tokens']>cfg['max_total_tokens']:bad.append('too_long')
            if bad:continue
            seen.add(key);row['language']='en';row['review']={'status':'accepted',
                'basis':'SmolTalk2 curated source plus exhaustive local row audit',
                'checks':['english_dominant','not_math','not_high_risk','not_time_sensitive',
                          'not_advertising','self_contained','no_repeat','length_ok','protocol_safe']}
            candidates.append(row)
            if len(candidates)==need:break
        if len(candidates)!=need:raise RuntimeError(f'English {name}: need {need}, accepted {len(candidates)}')
        rows.extend(candidates)
    if len(rows)!=256:raise RuntimeError(f'expected 256 English rows, found {len(rows)}')
    random.Random(cfg['seed']+17).shuffle(rows)
    return rows
def filter_and_review(root_cfg,output):
    from transformers import AutoTokenizer
    cfg=root_cfg['dataset']['general_sft'];system=root_cfg['platform_protocol']['system_prompt']
    raw_manifest=read_json(output/'zh_raw_manifest.json')
    teacher=path(root_cfg['teacher']['path'])
    tok=AutoTokenizer.from_pretrained(str(teacher),trust_remote_code=True,local_files_only=True)
    seen=set();zh_rows=[];review_stats={}
    for component in cfg['chinese']['components']:
        name=component['config'];rec=raw_manifest['files'][name];file=Path(rec['path'])
        if sha256(file)!=rec['sha256']:raise RuntimeError(f'raw checksum mismatch: {file}')
        accepted=[];rejects={}
        order=read_jsonl(file);random.Random(cfg['seed']+31+sum(map(ord,name))).shuffle(order)
        for row in order:
            value,reasons=audit_chinese(row,name,cfg,tok,system,seen)
            if value is not None:accepted.append(value)
            else:
                for reason in reasons:rejects[reason]=rejects.get(reason,0)+1
            if len(accepted)==component['target_rows']:break
        if len(accepted)!=component['target_rows']:
            raise RuntimeError(f'{name}: need {component["target_rows"]}, accepted {len(accepted)}; rejects={rejects}')
        zh_rows.extend(accepted);review_stats[name]={'accepted':len(accepted),'reject_counts':rejects}
    if len(zh_rows)!=512:raise RuntimeError('Chinese pool must contain 512 rows')
    random.Random(cfg['seed']+43).shuffle(zh_rows)
    en_rows=build_english(cfg,system)
    zh_train,zh_eval=zh_rows[:448],zh_rows[448:]
    en_train,en_eval=en_rows[:224],en_rows[224:]
    manifest={'schema_version':2,'platform_protocol':root_cfg['platform_protocol'],
              'language_ratio':{'chinese':512,'english':256,'ratio':'2:1'},
              'review_policy':{'every_final_row_checked':True,
                'meaning':'source-level human/curation evidence plus deterministic row-level content, safety, repetition, length and protocol checks',
                'semantic_fact_check_limit':'not an external expert fact-check'},
              'chinese_source':raw_manifest,'review_stats':review_stats,'files':{}}
    for key,values in (('chinese_train',zh_train),('chinese_eval',zh_eval),('english_train',en_train),('english_eval',en_eval)):
        file=output/f'{key}.jsonl';write_jsonl(file,values)
        manifest['files'][key]={'path':str(file),'rows':len(values),'sha256':sha256(file),
            'min_tokens':min(x['total_tokens'] for x in values),'max_tokens':max(x['total_tokens'] for x in values),
            'mean_tokens':sum(x['total_tokens'] for x in values)/len(values)}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(path('configs/opd_hybrid_v3.json')))
    mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--download-chinese',action='store_true');mode.add_argument('--filter-review',action='store_true')
    args=ap.parse_args();root=read_json(args.config);cfg=root['dataset']['general_sft'];output=path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    if args.download_chinese:download_chinese(cfg,output)
    else:filter_and_review(root,output)
if __name__=='__main__':main()



