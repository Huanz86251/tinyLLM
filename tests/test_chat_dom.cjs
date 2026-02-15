const fs=require("node:fs");
const path=require("node:path");
const assert=require("node:assert/strict");
const {randomUUID}=require("node:crypto");
const {JSDOM}=require("./dom/node_modules/jsdom");
const root=path.resolve(__dirname,"..");
const dom=new JSDOM(fs.readFileSync(root+"/ui/index.html","utf8"),{url:process.env.TINYLLM_TEST_URL||"http://127.0.0.1:8502/",runScripts:"outside-only",pretendToBeVisual:true});
const w=dom.window;
w.fetch=(url,opts)=>fetch(new URL(url,w.location.href),opts);
w.TextDecoder=TextDecoder;
w.crypto.randomUUID=randomUUID;
w.HTMLElement.prototype.scrollTo=function(){};
w.HTMLDialogElement.prototype.showModal=function(){this.setAttribute("open","");};
w.HTMLDialogElement.prototype.close=function(){this.removeAttribute("open");};
w.navigator.clipboard={writeText:async()=>{}};
const errors=[];w.addEventListener("error",e=>errors.push(e.message));
const checks=[];
function check(name,ok,detail){checks.push({name,passed:!!ok,detail});console.log(name,ok?"PASS":"FAIL",detail||"");assert.ok(ok,name);}
async function waitFor(fn,ms=120000){const start=Date.now();while(!fn()){if(Date.now()-start>ms)throw Error("Timed out: "+fn.toString());await new Promise(r=>setTimeout(r,100));}}
async function clickSend(){await waitFor(()=>!get("sendButton").disabled);get("sendButton").click();}
function input(text){const e=w.document.getElementById("prompt");e.value=text;e.dispatchEvent(new w.Event("input",{bubbles:true}));}
const get=id=>w.document.getElementById(id);
for (const file of ["/ui/vendor/katex/katex.min.js","/ui/rendering.js","/ui/app.js"]) require("node:vm").runInContext(fs.readFileSync(root+file,"utf8"),dom.getInternalVMContext());
(async()=>{
try{
 await waitFor(()=>get("globalStatus").textContent.includes("就绪"));
 check("initial mode is SFT chat",get("modeLabel").textContent==="普通聊天");
 check("empty send disabled",get("sendButton").disabled);
 get("modeTrigger").click();
 check("bottom mode selector opens",!get("modeMenu").classList.contains("hidden"));
 check("OPD mode is available",w.document.querySelector('[data-mode="opd"]')!==null);
 w.document.querySelector('[data-mode="opd"]').click();
 await waitFor(()=>get("modeLabel").textContent==="OPD 数学"&&!w.eval("selecting")&&!w.eval("serverBusy"));
 check("OPD mode identifies its teacher LoRA",get("composerNote").textContent.includes("MiniCPM3-4B OPD LoRA"));
 get("modeTrigger").click();
 w.document.querySelector('[data-mode="math"]').click();
 await waitFor(()=>get("modeLabel").textContent==="数学推理"&&!w.eval("selecting")&&!w.eval("serverBusy"));
 check("plain math shows verified strict protocol",get("composerNote").textContent.includes("严格贪心")&&get("composerNote").textContent.includes("640"));
 input("What is 12 minus 5?");
 const rendered=[];
 const observer=new w.MutationObserver(()=>{const text=w.document.querySelector(".assistant-body")?.textContent;if(text)rendered.push(text);});
 observer.observe(get("messages"),{childList:true,subtree:true,characterData:true});
 await clickSend();
 await waitFor(()=>w.eval("generating"));
 check("send becomes stop during generation",get("sendButton").getAttribute("aria-label")==="停止生成");
 await waitFor(()=>!w.eval("generating"));
 observer.disconnect();
 check("token stream rendered multiple times",new Set(rendered).size>5,new Set(rendered).size);
 check("math response shown",get("messages").textContent.includes("7"));
 check("stream metrics displayed",get("messages").textContent.includes("token/s"));
 check("history list contains conversation",get("chatList").querySelectorAll("button").length===1);
 // Upload a supplied image through the same browser-facing control flow.
 get("attachButton").click();get("exampleCat").click();
 await waitFor(()=>get("modeLabel").textContent==="图片问答"&&!w.eval("selecting")&&!w.eval("serverBusy"));
 check("image auto-switches to VLM",get("attachmentPreview").querySelector("img")!==null);
 input("Describe the animal in this picture.");await clickSend();
 await waitFor(()=>w.eval("generating"));await waitFor(()=>!w.eval("generating"));
 check("image answer reaches chat",get("messages").lastElementChild.textContent.includes("cat"));
 check("image follow-up context remains visible",!get("contextImage").classList.contains("hidden"));
 input("What color is its fur?");await clickSend();
 await waitFor(()=>w.eval("generating"));await waitFor(()=>!w.eval("generating"));
 check("image feature reuse visible",get("messages").lastElementChild.textContent.includes("图像特征已复用"));
 get("newChat").click();
 await waitFor(()=>get("modeLabel").textContent==="普通聊天"&&!w.eval("selecting")&&!w.eval("serverBusy"));
 check("new chat clears image context",get("contextImage").classList.contains("hidden"));
 check("prior chat remains in sidebar",get("chatList").querySelectorAll("button").length===1);
 input("Write a long explanation of how computers work.");await clickSend();
 await waitFor(()=>w.eval("generating")&&w.document.querySelector(".assistant-body")?.textContent.includes("computer"));
 await clickSend();await waitFor(()=>!w.eval("generating"));
 check("stop button cancels actual generation",get("messages").lastElementChild.textContent.includes("已停止"));
 const dangerous=w.markdown('<img src=x onerror="alert(1)"> <script>alert(1)</script>');
 const detached=w.document.createElement("div");detached.innerHTML=dangerous;
 check("model markup cannot execute HTML",!detached.querySelector("img,script"));
 get("aboutOpen").click();check("model provenance dialog opens",get("aboutDialog").hasAttribute("open"));
 check("OPD subset result is labelled separately from the full benchmark",get("aboutDialog").textContent.includes("49.51%")&&get("aboutDialog").textContent.includes("50.00%")&&get("aboutDialog").textContent.includes("不是 GSM8K 全量分数"));

 const protectedNode=w.assistantNode({mode:"vision",content:"保留的描述。",stats:{stop_reason:"repetition",tokens:90,decoding:{}}});
 check("repetition stop warning is visible",protectedNode.querySelector(".decoding-notice")?.textContent.includes("循环重复"));
 check("repetition stop is not presented as complete answer",protectedNode.textContent.includes("内容可能不完整"));
 const formulaNode=w.assistantNode({mode:"math",content:"<|thought_start|>Equation: \\(x^2+1=y\\)<|thought_end|>Answer: $\\boxed{7}$",stats:{stop_reason:"boxed",tokens:100}});
 check("both thought and final formulas rendered",formulaNode.querySelector(".thought-text .katex") && formulaNode.querySelectorAll(".katex").length===2);
 check("no JavaScript runtime errors",errors.length===0,errors);
}finally{
 fs.writeFileSync(root+"/logs/history/chat_dom_check.json",JSON.stringify({checks,errors,method:"JSDOM interactions against real local inference service",visual_browser_check:false},null,2));
 dom.window.close();
}
})().catch(e=>{console.error(e);process.exitCode=1;});
