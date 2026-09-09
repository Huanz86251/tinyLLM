"""Staged single-GPU OPD for MiniCPM3-4B -> custom 0.51B tinyLLM."""
import argparse,gc,json,math,random,re,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from local_datasets import load_local_dataset
from train import GRPO as g
from project_paths import path

LOOP_RE=re.compile(r"(?P<unit>.{16,160}?)(?P=unit){2,}",re.S)

def is_loop(text): return bool(LOOP_RE.search(text or ""))

def generalized_jsd(student_logits, teacher_logits, beta=0.5, temperature=1.0,
                    token_mask=None):
    """Generalized JSD with an optional [batch, token] semantic-token mask."""
    s_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1).detach()
    mix = torch.logaddexp(s_log + math.log(1 - beta), t_log + math.log(beta))
    left = (s_log.exp() * (s_log - mix)).sum(-1)
    right = (t_log.exp() * (t_log - mix)).sum(-1)
    per_token = ((1 - beta) * left + beta * right) * (temperature ** 2)
    if token_mask is None:
        return per_token.mean()
    mask = token_mask.to(device=per_token.device, dtype=per_token.dtype)
    denominator = mask.sum()
    if int(denominator.detach().item()) == 0:
        return student_logits.sum() * 0.0
    return (per_token * mask).sum() / denominator


def chunked_jsd(student_logits, teacher_logits, beta, temperature, chunk_tokens,
                token_mask=None):
    """Memory-bounded JSD; protocol tokens may be excluded without changing normalization."""
    tokens = student_logits.size(1)
    total = student_logits.new_zeros((), dtype=torch.float32)
    denominator = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, tokens, chunk_tokens):
        end = min(tokens, start + chunk_tokens)
        part_mask = None if token_mask is None else token_mask[:, start:end]
        if part_mask is None:
            count = end - start
        else:
            count = int(part_mask.sum().detach().item())
            if count == 0:
                continue
        part = generalized_jsd(student_logits[:, start:end], teacher_logits[:, start:end],
                               beta, temperature, token_mask=part_mask)
        total = total + part * count
        denominator = denominator + count
    if int(denominator.detach().item()) == 0:
        return student_logits.sum() * 0.0
    return total / denominator

class FP32MasterAdamW:
    """Keep tiny BF16 LoRA modules in-model while optimizer state and updates stay FP32."""
    def __init__(self,model_parameters,lr):
        self.model_parameters=list(model_parameters)
        self.master=[torch.nn.Parameter(p.detach().float().clone(),requires_grad=True) for p in self.model_parameters]
        self.optimizer=torch.optim.AdamW(self.master,lr=lr,weight_decay=0.0)
    def zero_grad(self):
        for p in self.model_parameters:p.grad=None
        self.optimizer.zero_grad(set_to_none=True)
    def set_lr(self,lr):
        for group in self.optimizer.param_groups:group["lr"]=lr
    def step(self):
        for source,master in zip(self.model_parameters,self.master):
            master.grad=None if source.grad is None else source.grad.detach().float().clone()
        self.optimizer.step()
        with torch.no_grad():
            for source,master in zip(self.model_parameters,self.master):source.copy_(master.to(source.dtype))

def learning_rate(step,total,peak,floor,warmup):
    if step<=warmup:return peak*step/max(warmup,1)
    progress=(step-warmup)/max(total-warmup,1)
    return floor+(peak-floor)*0.5*(1+math.cos(math.pi*min(progress,1.0)))

def evaluate_indices(model,tok,dataset,indices,max_new_tokens,batch_size):
    rows=[]; was_training=model.training; model.eval()
    for at in range(0,len(indices),batch_size):
        ids=indices[at:at+batch_size];batch=[dataset[i] for i in ids]
        prompts=[g.build_gsm8k_prompt(ex["question"]) for ex in batch]
        out=g.sample_for_gsm8k_eval_batch(model,tok,prompts,g.SYSTEM_PROMPT_FOR_COT,max_new_tokens=max_new_tokens)
        for index,ex,text,tokens in zip(ids,batch,out["texts"],out["token_ids"]):
            parsed=g.parse_gsm8k_prediction(text,tok);gt=g.extract_gsm8k_gt(ex["answer"])
            correct=parsed["final_answer"] is not None and gt is not None and abs(parsed["final_answer"]-gt)<1e-6
            rows.append({"index":index,"correct":bool(correct),"prediction":parsed["final_answer"],"ground_truth":gt,
                         "tokens":len(tokens),"loop":is_loop(text),"boxed":parsed["has_box_and_valid"]})
    if was_training:model.train()
    n=len(rows)
    return {"correct":sum(r["correct"] for r in rows),"total":n,"accuracy":sum(r["correct"] for r in rows)/n,
            "loops":sum(r["loop"] for r in rows),"loop_rate":sum(r["loop"] for r in rows)/n,
            "boxed_rate":sum(r["boxed"] for r in rows)/n,"average_tokens":sum(r["tokens"] for r in rows)/n,"rows":rows}

def save_json(file,data):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True);tmp=file.with_suffix(file.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(file)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default=str(path("configs/opd_gsm8k_stage.json")))
    ap.add_argument("--trajectories",type=int);ap.add_argument("--max-new-tokens",type=int);ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text(encoding="utf-8-sig"));tc=dict(cfg["training"])
    if args.trajectories:tc["trajectories"]=args.trajectories
    if args.max_new_tokens:tc["rollout_max_new_tokens"]=args.max_new_tokens
    if args.smoke:tc["trajectories"]=1;tc["gradient_accumulation"]=1
    random.seed(tc["seed"]);torch.manual_seed(tc["seed"]);torch.cuda.manual_seed_all(tc["seed"])
    stamp=time.strftime("%Y%m%d-%H%M%S");run_dir=path("runs/training/opd_gsm8k_stage")/stamp;run_dir.mkdir(parents=True)
    report={"status":"starting","started_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"config":cfg,"effective_training":tc,
            "run_dir":str(run_dir),"smoke":args.smoke,"history":[],"evaluations":[]}
    report_path=run_dir/"report.json";save_json(report_path,report)
    try:
        train_dataset=load_local_dataset("gsm8k","train");eval_dataset=load_local_dataset("gsm8k","test")
        order=list(range(len(train_dataset)));random.Random(cfg["dataset"]["split_seed"]).shuffle(order)
        validation_indices=list(range(cfg["dataset"]["validation_start"],cfg["dataset"]["validation_start"]+cfg["dataset"]["validation_size"]))
        training_indices=order[:tc["trajectories"]]
        if len(training_indices)<tc["trajectories"]:raise RuntimeError("not enough unique GSM8K prompts")
        if validation_indices[-1]>=len(eval_dataset):raise RuntimeError("validation slice exceeds GSM8K test")
        report["split"]={"validation_source":"test","validation_indices":validation_indices,"training_source":"train","training_indices":training_indices,"cross_split":True}
        teacher_dir=path(cfg["teacher"]["path"]);tok=AutoTokenizer.from_pretrained(str(teacher_dir),trust_remote_code=True,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        tok.padding_side="left"
        student=g.load_tinyllm_from_ckpt(str(path(cfg["student"]["path"])),tok,strict=True)
        sc=cfg["student"];student.attach_lora_adapter(sc["adapter_name"],rank=sc["lora_rank"],dropout=0.0,alpha=sc["lora_alpha"],target=sc["lora_target"])
        student.activate_single_lora(sc["adapter_name"])
        for name,p in student.named_parameters():p.requires_grad_(f".adapters.{sc['adapter_name']}." in name)
        trainable=[p for p in student.parameters() if p.requires_grad];report["trainable_parameters"]=sum(p.numel() for p in trainable)
        if not args.smoke:
            baseline=evaluate_indices(student,tok,eval_dataset,validation_indices,cfg["evaluation"]["max_new_tokens"],cfg["evaluation"]["batch_size"])
            report["baseline_validation"]=baseline;print(f"baseline {baseline['correct']}/{baseline['total']} loop={baseline['loops']}",flush=True)
        teacher=AutoModelForCausalLM.from_pretrained(str(teacher_dir),trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to("cuda").eval()
        for p in teacher.parameters():p.requires_grad_(False)
        optimizer=FP32MasterAdamW(trainable,tc["peak_learning_rate"]);optimizer.zero_grad()
        accum=tc["gradient_accumulation"];total_updates=math.ceil(tc["trajectories"]/accum);update=0
        eval_points=set(cfg["evaluation"]["after_trajectories"]);best=None;began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for trajectory,(question_index) in enumerate(training_indices,start=1):
            ex=train_dataset[question_index];user=g.build_gsm8k_prompt(ex["question"])
            sample=g.sample_for_grpo_manual(student,tok,user,g.SYSTEM_PROMPT_FOR_COT,num_generations=1,max_new_tokens=tc["rollout_max_new_tokens"],
                temperature=tc["temperature"],top_p=tc["top_p"],repetition_penalty=tc["repetition_penalty"],no_repeat_ngram_size=tc["no_repeat_ngram_size"])
            generated=sample["token_ids"][0];text=sample["texts"][0]
            if not generated:raise RuntimeError("student produced an empty trajectory")
            prompt=g.render_prompt_with_tokenizer(tok,user,g.SYSTEM_PROMPT_FOR_COT);context=tok.encode(prompt,add_special_tokens=False)
            full=torch.tensor([context+generated],dtype=torch.long,device="cuda");mask=torch.ones_like(full);start=len(context)-1;end=start+len(generated)
            with torch.no_grad():
                teacher_out=teacher(input_ids=full,attention_mask=mask,use_cache=False)
                teacher_logits=teacher_out.logits[:,start:end,:].detach()
            student.train();student_out=student(input_ids=full,attention_mask=mask,use_cache=False,force_checkpoint=False)
            student_logits=student_out["logits"][:,start:end,:]
            loss=chunked_jsd(student_logits,teacher_logits,tc["jsd_beta"],tc["distillation_temperature"],tc["jsd_chunk_tokens"])
            if not torch.isfinite(loss):raise RuntimeError("non-finite OPD loss")
            (loss/accum).backward()
            capped=len(generated)>=tc["rollout_max_new_tokens"]
            record={"trajectory":trajectory,"question_index":question_index,"tokens":len(generated),"capped":capped,
                    "boxed":bool(g.BOX_RE.search(text)),"loop":is_loop(text),"loss":float(loss.detach()),"optimizer_step":None}
            boundary=(trajectory%accum==0 or trajectory==len(training_indices))
            if boundary:
                update+=1;grad=float(torch.nn.utils.clip_grad_norm_(trainable,tc["max_grad_norm"]));lr=learning_rate(update,total_updates,tc["peak_learning_rate"],tc["minimum_learning_rate"],tc["warmup_optimizer_steps"])
                optimizer.set_lr(lr);optimizer.step();optimizer.zero_grad();record.update(optimizer_step=update,grad_norm=grad,learning_rate=lr)
            report["history"].append(record)
            print(f"trajectory {trajectory}/{len(training_indices)} tok={len(generated)} cap={int(capped)} loss={record['loss']:.6f}"+(f" update={update}" if boundary else ""),flush=True)
            del teacher_out,teacher_logits,student_out,student_logits,loss,full,mask
            if (not args.smoke) and trajectory in eval_points:
                result=evaluate_indices(student,tok,eval_dataset,validation_indices,cfg["evaluation"]["max_new_tokens"],cfg["evaluation"]["batch_size"])
                adapter=run_dir/f"adapter_after_{trajectory}.pt";torch.save(student.get_lora_state_dict(sc["adapter_name"]),adapter)
                item={"after_trajectories":trajectory,"optimizer_steps":update,"adapter":str(adapter),**result};report["evaluations"].append(item)
                base=report["baseline_validation"];eligible=result["correct"]>base["correct"] and result["loop_rate"]<=base["loop_rate"]
                item["eligible_vs_baseline"]=eligible
                if eligible and (best is None or result["correct"]>best["correct"] or (result["correct"]==best["correct"] and result["loop_rate"]<best["loop_rate"])):best=item
                print(f"validation@{trajectory} {result['correct']}/{result['total']} loop={result['loops']} eligible={eligible}",flush=True)
            report["peak_gpu_mb"]=torch.cuda.max_memory_allocated()/2**20;report["elapsed_seconds"]=time.perf_counter()-began;save_json(report_path,report)
        report["status"]="completed";report["best_candidate"]=best;report["eligible_for_demo"]=best is not None;report["auto_activated"]=False
        report["trajectory_summary"]={"total":len(report["history"]),"capped":sum(x["capped"] for x in report["history"]),"boxed":sum(x["boxed"] for x in report["history"]),"loops":sum(x["loop"] for x in report["history"])}
        report["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%S");save_json(report_path,report);save_json(path("logs/history/opd_gsm8k_stage_latest.json"),report)
        print(json.dumps({"status":report["status"],"peak_gpu_mb":report["peak_gpu_mb"],"trajectory_summary":report["trajectory_summary"],"baseline":report.get("baseline_validation"),"evaluations":[{k:v for k,v in x.items() if k!="rows"} for x in report["evaluations"]],"best":best},ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        report["status"]="failed";report["error"]=str(exc);report["traceback"]=traceback.format_exc();report["peak_gpu_mb"]=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0;save_json(report_path,report);raise

if __name__=="__main__":main()
