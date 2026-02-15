import argparse,json
from inference.runtime import load_model,generate_batch,log_result,registry,path

def main():
    p=argparse.ArgumentParser(description='TinyLLM local inference; never starts training.')
    p.add_argument('--model',choices=list(registry()),default=None)
    p.add_argument('--mode',choices=['math','chat'],default='math',
                   help='math reproduces the verified GSM8K baseline protocol')
    p.add_argument('--prompt',default='A box has 12 pencils. I give away 5 pencils. How many pencils remain?')
    p.add_argument('--image',help='Local image file; requires --model vlm')
    p.add_argument('--max-new-tokens',type=int,default=None)
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--system',default=None)
    args=p.parse_args()
    model_id=args.model or 'sft_base'
    max_new_tokens=args.max_new_tokens or (640 if args.mode=='math' else 2048)
    prompt=args.prompt
    system=args.system
    protocol=None
    model_record=registry()[model_id]
    is_vlm='vision_tower' in model_record
    if args.mode=='math' and not is_vlm:
        rec=json.loads(path('configs/prompts.json').read_text(encoding='utf-8'))['gsm8k']
        system=system or rec['system']
        prompt=prompt.rstrip()+rec['suffix']
        protocol='verified GSM8K baseline: SFT 50000, greedy, max_new_tokens=640'
    if is_vlm:
        if not args.image:p.error('the selected VLM requires --image')
        from inference.vision import answer_image
        result=answer_image(args.image,prompt,max_new_tokens,args.precision,model_id)
    else:
        if args.image:p.error('--image requires a model with a vision tower')
        model,tok,info=load_model(model_id,args.precision)
        result={**info,'mode':args.mode,'protocol':protocol,'prompt':prompt,'system':system,
                'max_new_tokens':max_new_tokens,
                **generate_batch(model,tok,[prompt],system,max_new_tokens)}
    output=log_result('demo',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Saved:',output)
if __name__=='__main__':main()
