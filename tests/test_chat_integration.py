"""Run against the local chat service; records real GPU behavior, not fabricated replies."""
import json,time,uuid,urllib.request,urllib.error,os
from pathlib import Path

BASE=os.environ.get("TINYLLM_TEST_URL","http://127.0.0.1:8502")
report={"checks":[],"conversations":[],"visual_browser_check":False}
def call(endpoint,data=None):
    req=urllib.request.Request(BASE+endpoint,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type":"application/json"} if data is not None else {})
    with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
def check(name,condition,detail=None):
    report["checks"].append({"name":name,"passed":bool(condition),"detail":detail})
    print(name,"PASS" if condition else "FAIL",detail or "",flush=True)
    if not condition:raise AssertionError(name)
def run(mode,text,*,image=None,messages=None,tokens=64,cancel_after=None):
    job=str(uuid.uuid4());start=time.perf_counter();events=[];token_arrivals=[]
    data={"request_id":job,"mode":mode,"messages":messages or [{"role":"user","content":text}],"max_tokens":tokens,"image_id":image}
    request=urllib.request.Request(BASE+"/api/chat",data=json.dumps(data).encode(),headers={"Content-Type":"application/json"})
    did_cancel=False
    with urllib.request.urlopen(request,timeout=180) as response:
        for line in response:
            event=json.loads(line)
            events.append(event)
            if event["type"]=="token":
                token_arrivals.append(time.perf_counter()-start)
                if cancel_after and len(token_arrivals)>=cancel_after and not did_cancel:
                    check("cancel request accepted",call("/api/cancel",{"request_id":job})["cancelled"])
                    did_cancel=True
    end=time.perf_counter()-start
    done=events[-1]
    check(mode+" completed without runtime error",done["type"]=="done",done.get("message"))
    row={"mode":mode,"done":done,"arrival_times":token_arrivals,"wall_seconds":end,
         "statuses":[e["message"] for e in events if e["type"]=="status"],
         "context":next((e for e in events if e["type"]=="context"),None)}
    report["conversations"].append(row)
    print(mode,repr(done.get("text",""))[:180],flush=True)
    return done,row

try:
    state=call("/api/status")
    check("default SFT preloaded",state["loaded"] and state["mode"]=="chat",state)
    baseline,_=run("chat","A box has 12 pencils. I give away 5. How many remain? Give a short answer.",tokens=64)
    arc,_=run("arc","Which material is attracted to a magnet?\nA. Wood\nB. Iron\nC. Plastic\nD. Glass",tokens=80)
    back,row=run("chat","A box has 12 pencils. I give away 5. How many remain? Give a short answer.",tokens=64)
    check("SFT ARC SFT reuse same base instance",baseline["base_instance"]==arc["base_instance"]==back["base_instance"])
    check("LoRA disabled restores identical base answer",baseline["text"]==back["text"])
    check("text modes did not reload base",baseline["base_loads"]==arc["base_loads"]==back["base_loads"])
    check("tokens arrive before completion",len(row["arrival_times"])>=3 and row["arrival_times"][0]<row["wall_seconds"]-.03,row["arrival_times"][:3])
    cancelled,row=run("chat","Write a long explanation of how computers work.",tokens=512,cancel_after=4)
    check("generation stopped early",cancelled["cancelled"] and cancelled["tokens"]<512,cancelled["tokens"])
    image=call("/api/example",{"name":"cat"})["image_id"]
    vision,row=run("chat","Describe the animal in this picture.",image=image,tokens=64)
    check("upload routes automatically to vision",vision["mode"]=="vision")
    check("vision family replaced text family",vision["base_loads"]==back["base_loads"]+1,call("/api/status"))
    follow_messages=[{"role":"user","content":"Describe the animal in this picture."},
                     {"role":"assistant","content":vision["text"] or "The image is shown."},
                     {"role":"user","content":"What color is its fur?"}]
    follow,row=run("vision","",image=image,messages=follow_messages,tokens=64)
    check("same image features reused",follow["image_cached"])
    check("vision follow-up retained model",vision["base_instance"]==follow["base_instance"])
    saved=json.loads(Path(follow["log"]).read_text(encoding="utf-8"))
    check("multiturn context contains both questions","Describe the animal" in saved["prepared_prompt"] and "What color" in saved["prepared_prompt"])
    check("only one image marker group in multi-turn",saved["prepared_prompt"].count("<img>")==5)
    restored,row=run("math","What is 12 minus 5?",tokens=96)
    check("return from vision reloads only text",restored["base_loads"]==vision["base_loads"]+1)
    state=call("/api/status")
    check("VRAM back near text baseline",state["gpu_allocated_mb"]<2600,state["gpu_allocated_mb"])
    call("/api/unload",{})
    state=call("/api/status")
    check("explicit unload frees model GPU tensors",state["gpu_allocated_mb"]<100,state)
    call("/api/model",{"mode":"chat"})
    check("reload after unload",call("/api/status")["loaded"])
finally:
    report["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%S")
    output = Path(__file__).resolve().parents[1] / "logs" / "history" / "chat_integration_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

