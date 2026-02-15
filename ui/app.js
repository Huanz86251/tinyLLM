"use strict";
const $ = (id) => document.getElementById(id);
const modes = {
  chat:{label:"普通聊天",icon:"✦",model:"SFT · 0.5B",note:"SFT 基础模型 · 无 LoRA",maxTokens:2048},
  math:{label:"数学推理",icon:"∑",model:"SFT · GSM8K 50.64%",note:"SFT 50000 · 严格贪心 · 640 tokens",maxTokens:640},
  opd:{label:"OPD 数学",icon:"◎",model:"OPD LoRA",note:"SFT + MiniCPM3-4B OPD LoRA",maxTokens:640},
  arc:{label:"科学 · ARC",icon:"⌘",model:"ARC LoRA",note:"SFT + ARC 峰值 LoRA",maxTokens:2048},
  vision:{label:"图片问答",icon:"▧",model:"VLM v5 · Best 18000",note:"Continual v5 最佳 checkpoint + InternViT",maxTokens:2048}
};
function isVisionMode(value) {return value==="vision";}
let stored;
try {stored = JSON.parse(localStorage.getItem("tinyllm-chats-v2") || "[]");} catch {stored=[];}
let chats = Array.isArray(stored) ? stored.filter(c=>c && typeof c.id==="string" && Array.isArray(c.messages)).slice(0,30) : [];
let activeId = chats[0]?.id || null;
let mode = "chat", pendingImage = null, generating = false, selecting = false, serverBusy = true;
let jobId = null, toastTimer = null, currentStatus = null, currentResponse = null;
function chat() {return chats.find(c=>c.id===activeId);}
function persist() {try {localStorage.setItem("tinyllm-chats-v2", JSON.stringify(chats.slice(0,30)));}catch {toast("本机浏览器存储已满，本次对话仍可继续。");}}
function createChat() {
  const c={id:crypto.randomUUID(), title:"新对话", messages:[], mode:"chat", activeImage:null};
  chats.unshift(c);activeId=c.id;pendingImage=null;setModeUI("chat");render();return c;
}
function toast(text) {$("toast").textContent=text;$("toast").classList.remove("hidden");clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("toast").classList.add("hidden"),4200);}
function escapeHtml(text) {return String(text).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function markdown(raw) {return window.TinyLLMRender.markdown(raw);}
function scheduleTokenRender(node,message){
  const elapsed=performance.now()-(node._lastRenderAt||0);
  if(elapsed>=70){clearTimeout(node._renderTimer);updateAssistant(node,message,true);}
  else if(!node._renderTimer){node._renderTimer=setTimeout(()=>{node._renderTimer=null;updateAssistant(node,message,true);},70-elapsed);}
}

function safeImageUrl(img) {return img && /^[a-f0-9]{32}\.(png|jpg|webp)$/.test(img.image_id) ? "/api/images/"+img.image_id : null;}
function getImage() {return pendingImage || (isVisionMode(mode) ? chat()?.activeImage : null);}
function closeMenus() {
  $("attachMenu").classList.add("hidden");$("modeMenu").classList.add("hidden");
  $("attachButton").setAttribute("aria-expanded","false");$("modeTrigger").setAttribute("aria-expanded","false");
}
function setModeUI(next) {
  mode=modes[next]?next:"chat";const m=modes[mode];
  $("topMode").textContent=m.label;$("topModel").textContent=m.model;
  $("modeLabel").textContent=m.label;$("modeIcon").textContent=m.icon;$("composerNote").textContent=m.note;
  $("maxTokens").value=String(m.maxTokens);
  $("prompt").placeholder=isVisionMode(mode) ? "针对这张图片，想问什么？" : mode==="arc" ? "输入科学问题，或粘贴 ARC 题目和选项…" : (mode==="math"||mode==="opd") ? "输入一道数学题，我们一步步来…" : "发消息给 tinyLLM…";
  document.querySelectorAll("[data-mode]").forEach(b=>{const selected=b.dataset.mode===mode;b.setAttribute("aria-checked",String(selected));b.querySelector("i").textContent=selected?"✓":"";});
  updateAttachment();updateSend();
}
function updateSend() {
  const send=$("sendButton");
  send.classList.toggle("stopping",generating);send.firstElementChild.textContent=generating?"■":"↑";
  send.setAttribute("aria-label",generating?"停止生成":"发送消息");
  send.disabled=generating?false:(selecting||serverBusy||!$("prompt").value.trim());
  $("modeTrigger").disabled=generating||selecting;
  $("attachButton").disabled=generating||selecting;
  $("newChat").disabled=generating||selecting;
  $("unloadButton").disabled=generating||selecting||serverBusy;
}
async function api(url,data) {
  const response=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
  const body=await response.json();
  if(!response.ok)throw new Error(body.error || "请求未完成");
  return body;
}
async function refreshStatus() {
  try {
    const response=await fetch("/api/status");if(!response.ok)throw new Error("连接失败");
    const state=await response.json();currentStatus=state;serverBusy=state.busy;
    document.querySelectorAll("[data-mode]").forEach(button=>{
      const ready=state.mode_available?.[button.dataset.mode]!==false;
      button.disabled=!ready;
      if(!ready)button.title="checkpoint 正在下载，完成后自动可选";else button.removeAttribute("title");
    });
    $("statusDot").className="online-dot"+(state.error?" error":state.busy?" waiting":"");
    $("globalStatus").textContent=state.busy?state.phase:state.loaded?"本地就绪 · "+(state.gpu_allocated_mb/1024).toFixed(1)+" GB":"模型未加载";
    $("deviceLabel").textContent=state.device || "本地设备";
    const progress=state.busy&&!generating;
    $("modelProgress").classList.toggle("hidden",!progress);
    $("progressText").textContent=state.phase;
    if(!generating) {
      $("warmHint").innerHTML=state.busy?'<span class="tiny-spinner"></span><span>'+escapeHtml(state.phase)+"，你可以先输入问题。</span>":state.error?'<span>'+escapeHtml(state.error)+"</span>":'<span>◈ '+(state.loaded?"模型已准备好 · 首次图片问答会自动加载视觉组件":"发送前会自动加载所选模型")+"</span>";
    }
    updateSend();
  }catch {
    serverBusy=true;$("globalStatus").textContent="本地服务未连接";$("statusDot").className="online-dot error";
    $("warmHint").textContent="请运行 start_demo.cmd，再刷新此页面。";updateSend();
  }
}
async function chooseMode(next, {preserveImage=false}={}) {
  if(generating||selecting)return;
  closeMenus();
  if(!preserveImage && !isVisionMode(next)) {pendingImage=null;if(chat())chat().activeImage=null;}
  setModeUI(next);
  if(chat())chat().mode=mode;
  persist();selecting=true;$("modelProgress").classList.remove("hidden");
  $("progressText").textContent=next==="arc"?"正在切换 ARC LoRA…":next==="opd"?"正在热加载 OPD 数学 LoRA…":isVisionMode(next)?"正在准备视觉模型…":"正在准备 SFT…";updateSend();
  try {
    // An initial warmup may still own the model; retry only after it becomes idle.
    let switched=false;
    for(let attempt=0;attempt<40;attempt++) {
      if(currentStatus?.busy&&!generating) {
        await new Promise(resolve=>setTimeout(resolve,350));
        await refreshStatus();continue;
      }
      await api("/api/model",{mode:next});switched=true;break;
    }
    if(!switched)toast("模型仍在处理上一项任务，发送时会自动切换到所选模式。");
  }catch(e){toast(e.message);}finally{selecting=false;await refreshStatus();updateSend();}
}
function updateAttachment() {
  const preview=$("attachmentPreview"),context=$("contextImage");
  preview.replaceChildren();context.replaceChildren();
  const img=pendingImage;
  preview.classList.toggle("hidden",!img);
  if(img) {
    const pic=document.createElement("img");pic.src=safeImageUrl(img);pic.alt="待发送图片";
    const note=document.createElement("span");note.textContent="新图片已添加 · 将开始独立视觉上下文";
    const remove=document.createElement("button");remove.textContent="×";remove.setAttribute("aria-label","移除图片");remove.onclick=removeImage;
    preview.append(pic,note,remove);
  }
  const active=!pendingImage&&isVisionMode(mode)?chat()?.activeImage:null;
  context.classList.toggle("hidden",!active);
  if(active) {
    const pic=document.createElement("img");pic.src=safeImageUrl(active);pic.alt="当前图片";
    const label=document.createElement("span");label.textContent="继续讨论这张图片 · 复用视觉特征";
    const remove=document.createElement("button");remove.textContent="结束图片话题 ×";remove.onclick=removeImage;
    context.append(pic,label,remove);
  }
}
function removeImage() {
  if(generating)return;
  pendingImage=null;if(chat())chat().activeImage=null;
  updateAttachment();if(isVisionMode(mode))chooseMode("chat");persist();
}
async function setImage(image) {
  pendingImage=image;
  if(!chat())createChat();
  pendingImage=image;chat().activeImage=image;
  await chooseMode(isVisionMode(mode)?mode:"vision",{preserveImage:true});updateAttachment();$("prompt").focus();
}
async function exampleImage(name) {
  closeMenus();if(generating||selecting)return;
  try {
    const image=await api("/api/example",{name});await setImage(image);
    if(!$("prompt").value.trim()) $("prompt").value=name==="cat"?"Describe the animal in this picture.":"Describe this picture briefly.";
    autosize();updateSend();
  }catch(e){toast(e.message);}
}
function renderList() {
  const list=$("chatList");list.replaceChildren();
  const visible=chats.filter(c=>c.messages.length);
  if(!visible.length){const p=document.createElement("p");p.className="list-empty";p.textContent="从第一条消息开始，\\n对话会留在这里。".replace("\\n","\n");list.append(p);}
  for(const c of visible.slice(0,30)) {
    const b=document.createElement("button");b.textContent=c.title;b.className=c.id===activeId?"active":"";b.title=c.title;
    b.onclick=()=>{if(generating||selecting){toast("请先停止当前回答。");return;}activeId=c.id;pendingImage=null;setModeUI(c.mode||"chat");render();chooseMode(c.mode||"chat",{preserveImage:true});$("sidebar").classList.remove("open");};
    list.append(b);
  }
}
function responseMarkup(text, streaming=false) {
  const start="<|thought_start|>",end="<|thought_end|>";
  const pos=text.indexOf(start),endpos=text.indexOf(end);
  if(pos>=0) {
    const thought=text.slice(pos+start.length,endpos>=0?endpos:undefined);
    const final=endpos>=0?text.slice(endpos+end.length).trim():"";
    return '<details class="thought"'+(streaming||!final?" open":"")+"><summary>"+(endpos<0&&streaming?"正在推理":"查看模型推理过程")+'</summary><div class="thought-text">'+markdown(thought)+'</div></details>'+(final?markdown(final):"");
  }
  return markdown(text.replace(/<\|thought_(start|end)\|>/g,""));
}
function assistantNode(message) {
  const node=document.createElement("article");node.className="message assistant";
  const head=document.createElement("div");head.className="assistant-head";
  head.innerHTML='<img src="/assets/logo.svg" alt=""><b>tinyLLM</b><span>'+escapeHtml(modes[message.mode]?.model||"SFT")+"</span>";
  const body=document.createElement("div");body.className="assistant-body";
  const meta=document.createElement("div");meta.className="response-meta";
  node.append(head,body,meta);updateAssistant(node,message,false);return node;
}
function updateAssistant(node,message,live) {
  clearTimeout(node._renderTimer);node._renderTimer=null;node._lastRenderAt=performance.now();
  const body=node.querySelector(".assistant-body");
  if(message.content)body.innerHTML=responseMarkup(message.content,live);
  else body.innerHTML='<div class="answer-status">'+(live?'<span class="tiny-spinner"></span>':"")+'<span>'+escapeHtml(message.phase||(message.cancelled?"已停止":"等待回答"))+"</span></div>";
  if(message.error){const error=document.createElement("div");error.className="error-note";error.textContent=message.error;body.append(error);}
  if(message.stats?.stop_reason==="length"){const n=document.createElement("div");n.className="warning-note";n.textContent="已达到输出长度上限，可以继续追问。";body.append(n);}
  const protectedStop=message.stats?.stop_reason;
  if(["repetition","time_limit","invalid_logits"].includes(protectedStop)){
    const notice=document.createElement("div");notice.className="warning-note decoding-notice";
    notice.textContent=protectedStop==="repetition"?"检测到循环重复，已自动停止并收起重复尾部。内容可能不完整，请核对或换一种问法。":
      protectedStop==="time_limit"?"已达到本次生成时间上限，回答可能未完成，可以继续追问。":"检测到异常解码数值，已停止生成；没有用编造内容补齐。";
    body.append(notice);
  }
  window.TinyLLMRender.renderMath(body);
  const meta=node.querySelector(".response-meta");meta.replaceChildren();
  const stats=message.stats;
  if(stats){
    const bits=[];
    if(stats.first_token_total_seconds!=null)bits.push("首 token "+stats.first_token_total_seconds.toFixed(2)+" 秒");
    if(stats.tokens_per_second)bits.push(stats.tokens_per_second.toFixed(1)+" token/s");
    bits.push((stats.tokens||0)+" tokens");
    if(stats.cancelled)bits.push("已停止");
    if(stats.image_cached)bits.push("图像特征已复用");
    if(stats.decoding)bits.push("防重复保护已启用");
    const label=document.createElement("span");label.textContent=bits.join(" · ");meta.append(label);
  }
  if(!live&&message.content){
    const b=document.createElement("button");b.className="copy-btn";b.textContent="复制";
    b.onclick=()=>navigator.clipboard.writeText(message.content).then(()=>toast("已复制回答")).catch(()=>toast("浏览器未允许复制，请手动选择文字。"));meta.append(b);
  }
}
function userNode(message) {
  const node=document.createElement("article");node.className="message user";
  if(safeImageUrl(message.image)){const img=document.createElement("img");img.src=safeImageUrl(message.image);img.alt="用户上传图片";img.className="user-image";node.append(img);}
  const bubble=document.createElement("div");bubble.className="user-bubble";bubble.textContent=message.content;node.append(bubble);return node;
}
function render() {
  renderList();$("messages").replaceChildren();
  const c=chat();$("welcome").classList.toggle("hidden",!!c?.messages.length);
  if(c)for(const m of c.messages)$("messages").append(m.role==="user"?userNode(m):assistantNode(m));
  updateAttachment();scrollBottom();updateSend();
}
function nearBottom() {const c=$("conversation");return c.scrollHeight-c.scrollTop-c.clientHeight<160;}
function scrollBottom(){requestAnimationFrame(()=>$("conversation").scrollTo({top:$("conversation").scrollHeight,behavior:"instant"}));}
function autosize() {const t=$("prompt");t.style.height="auto";t.style.height=Math.min(t.scrollHeight,180)+"px";}
async function stopGeneration() {
  if(!generating||!jobId)return;
  $("sendButton").disabled=true;
  try {await api("/api/cancel",{request_id:jobId});if(currentResponse){currentResponse.message.phase="正在停止…";if(!currentResponse.message.content)updateAssistant(currentResponse.node,currentResponse.message,true);}}
  catch(e){toast(e.message);$("sendButton").disabled=false;}
}
async function send() {
  if(generating){await stopGeneration();return;}
  if(selecting||serverBusy)return;
  const content=$("prompt").value.trim();if(!content)return;
  if(isVisionMode(mode)&&!getImage()){toast("请点左侧 ＋ 添加图片。");$("attachMenu").classList.remove("hidden");return;}
  closeMenus();
  const c=chat()||createChat();c.mode=mode;
  const image=getImage();const outgoingImage=pendingImage;
  if(image)c.activeImage=image;
  const user={role:"user",content,mode,image:outgoingImage||null};
  c.messages.push(user);if(c.messages.filter(m=>m.role==="user").length===1)c.title=content.slice(0,30);
  let relevant=c.messages.filter(m=>(m.role==="user"||m.role==="assistant")&&m.content&&!m.error).slice(-40);
  if(outgoingImage){
    // A newly uploaded image is a hard context boundary. Keep old messages in
    // the browser transcript, but send only the new image question to the model.
    relevant=[user];
  }else if(image){
    // Follow-up on the same image: retain only that image's visual conversation.
    const lastImage=relevant.map((m,i)=>m.role==="user"&&m.image?.image_id===image.image_id?i:-1).filter(i=>i>=0).pop();
    if(lastImage!=null)relevant=relevant.slice(lastImage);
  }
  while(relevant.length>1&&relevant[0].role!=="user")relevant.shift();
  const messages=relevant.map(m=>({role:m.role,content:m.content}));
  const assistant={role:"assistant",content:"",mode,phase:"准备开始…"};
  c.messages.push(assistant);pendingImage=null;
  $("prompt").value="";autosize();render();persist();
  const node=$("messages").lastElementChild;
  generating=true;jobId=crypto.randomUUID();currentResponse={node,message:assistant};
  updateSend();updateAssistant(node,assistant,true);scrollBottom();
  let receivedFinal=false;
  try {
    const response=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({request_id:jobId,mode,messages,image_id:image?.image_id||null,max_tokens:Number($("maxTokens").value)})});
    if(!response.ok){const problem=await response.json();throw new Error(problem.error||"请求未完成");}
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer="";
    while(true){
      const {value,done}=await reader.read();
      buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
      let newline;
      while((newline=buffer.indexOf("\n"))>=0){
        const line=buffer.slice(0,newline);buffer=buffer.slice(newline+1);if(!line.trim())continue;
        const event=JSON.parse(line);const follow=nearBottom();
        if(event.type==="status"){assistant.phase=event.message;if(!assistant.content)updateAssistant(node,assistant,true);}
        if(event.type==="context"&&event.dropped_messages){toast("对话较长，本次已省略较早的 "+event.dropped_messages+" 条消息。");}
        if(event.type==="token"){
          assistant.content=event.text;scheduleTokenRender(node,assistant);
        }
        if(event.type==="guard"){assistant.phase=event.message;}
        if(event.type==="context"&&event.cleaned_history_messages){toast("已清理较早回答中的循环片段，避免影响本次回答。");}
        if(event.type==="done"){
          assistant.content=event.text;assistant.stats=event;assistant.cancelled=event.cancelled;assistant.phase=event.cancelled?"已停止":"本次没有生成文字";
          receivedFinal=true;updateAssistant(node,assistant,false);
        }
        if(event.type==="error")throw new Error(event.message);
        if(follow)scrollBottom();
      }
      if(done)break;
    }
    if(!receivedFinal)throw new Error("连接提前结束，已保留收到的部分回答。");
  }catch(e){assistant.error=e.message;updateAssistant(node,assistant,false);}
  finally{generating=false;serverBusy=false;currentResponse=null;jobId=null;updateSend();persist();renderList();await refreshStatus();updateSend();$("prompt").focus();}
}
$("prompt").addEventListener("input",()=>{autosize();updateSend();});
$("prompt").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey&&!e.isComposing){e.preventDefault();send();}});
$("sendButton").onclick=send;
$("modeTrigger").onclick=e=>{e.stopPropagation();const open=$("modeMenu").classList.contains("hidden");closeMenus();$("modeMenu").classList.toggle("hidden",!open);$("modeTrigger").setAttribute("aria-expanded",String(open));};
$("attachButton").onclick=e=>{e.stopPropagation();const open=$("attachMenu").classList.contains("hidden");closeMenus();$("attachMenu").classList.toggle("hidden",!open);$("attachButton").setAttribute("aria-expanded",String(open));};
document.querySelectorAll("[data-mode]").forEach(b=>b.onclick=()=>chooseMode(b.dataset.mode));
document.addEventListener("click",e=>{if(!e.target.closest(".mode-wrap,.attach-wrap"))closeMenus();});
$("uploadChoice").onclick=()=>{closeMenus();$("fileInput").click();};
$("fileInput").addEventListener("change",async e=>{
  const file=e.target.files?.[0];if(!file)return;
  if(file.size>10*1024*1024){toast("请选择小于 10 MB 的图片。");e.target.value="";return;}
  try {
    const encoded=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(",")[1]);r.onerror=reject;r.readAsDataURL(file);});
    const image=await api("/api/upload",{data:encoded});await setImage(image);
  }catch(err){toast(err.message||"图片读取失败");}finally{e.target.value="";}
});
$("exampleCat").onclick=()=>exampleImage("cat");$("exampleAstronaut").onclick=()=>exampleImage("astronaut");
$("newChat").onclick=()=>{if(generating||selecting)return;createChat();chooseMode("chat");$("prompt").focus();$("sidebar").classList.remove("open");};
$("sidebarToggle").onclick=()=>$("sidebar").classList.toggle("open");
$("aboutOpen").onclick=()=>$("aboutDialog").showModal();$("aboutClose").onclick=()=>$("aboutDialog").close();
$("unloadButton").onclick=async()=>{try{await api("/api/unload",{});await refreshStatus();toast("模型显存已释放，下次发送时会自动重新加载。");}catch(e){toast(e.message);}};
document.querySelectorAll("[data-suggest]").forEach(b=>b.onclick=async()=>{
  if(generating||selecting)return;
  const next=b.dataset.suggest;
  if(next==="vision"){await exampleImage("cat");return;}
  await chooseMode(next);
  $("prompt").value=next==="math"?"A box has 12 pencils. I give away 5 pencils. How many pencils remain?":"请用简单的语言解释什么是机器学习。";
  autosize();updateSend();$("prompt").focus();
});
document.addEventListener("keydown",e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="k"){e.preventDefault();$("newChat").click();}if(e.key==="Escape"){closeMenus();$("sidebar").classList.remove("open");}});
if(!activeId)createChat();else{setModeUI(chat()?.mode||"chat");render();}
refreshStatus();setInterval(refreshStatus,1200);

