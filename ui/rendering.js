/* Offline, escaped Markdown + KaTeX DOM rendering. Model HTML is never trusted. */
(() => {
  "use strict";
  const escapeHtml = value => String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  function markdown(raw) {
    const fragments=[];
    const hold=html=>{fragments.push(html);return "\u0000FRAGMENT"+(fragments.length-1)+"\u0000";};
    let source=String(raw).replace(/\u0000/g,"\ufffd");
    // Protect code before looking for math, including a still-open streamed fence.
    source=source.replace(/\x60\x60\x60[^\n]*\n([\s\S]*?)(?:\x60\x60\x60|$)/g,(_,code)=>hold("<pre><code>"+escapeHtml(code)+"</code></pre>"));
    source=source.replace(/\x60([^\x60\n]+)\x60/g,(_,code)=>hold("<code>"+escapeHtml(code)+"</code>"));
    // Normalize only escaped math delimiters, never all LaTeX backslashes.
    source=source.replace(/(?<!\\)\\\\([()[\]])/g,(_,delimiter)=>"\\"+delimiter);
    function math(tex,display,original){
      if(tex.length>3000)return hold('<span class="math-source">'+escapeHtml(original)+"</span>");
      // A doubled command slash is a frequent model artifact. Preserve \\ row breaks.
      tex=tex.replace(/(?<!\\)\\\\(?=[a-zA-Z])/g,"\\");
      return hold('<span class="math-pending '+(display?"math-display":"math-inline")+'" data-math-pending="1" data-display="'+(display?"1":"0")+'" data-tex="'+escapeHtml(tex)+'">'+escapeHtml(original)+"</span>");
    }
    source=source.replace(/\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|(?<!\\)\$(?!\$)([^\n$]+?)\$(?!\$)/g,
      (whole,dollars,brackets,parens,inline)=>{
        const tex=dollars??brackets??parens??inline;
        // Do not turn a currency phrase "$5 and $10" into an equation.
        if(inline && /^\d+(?:\.\d+)?\s+(?:and|or|到|和)\b/i.test(inline))return whole;
        return math(tex,dollars!==undefined||brackets!==undefined,whole);
      });
    // Standalone boxed answers, including nested fractions.
    let boxedAt=0;
    while((boxedAt=source.indexOf("\\boxed",boxedAt))>=0){
      const open=source.indexOf("{",boxedAt+6);
      if(open<0||!/^\s*$/.test(source.slice(boxedAt+6,open))){boxedAt+=6;continue;}
      let depth=1,at=open+1;
      for(;at<source.length&&depth;at++){
        if(source[at]==="{"&&source[at-1]!=="\\")depth++;
        if(source[at]==="}"&&source[at-1]!=="\\")depth--;
      }
      if(depth){boxedAt+=6;continue;}
      const whole=source.slice(boxedAt,at),placeholder=math(whole,false,whole);
      source=source.slice(0,boxedAt)+placeholder+source.slice(at);
      boxedAt+=placeholder.length;
    }
    let safe=escapeHtml(source).replace(/\*\*([^*\n]+)\*\*/g,"<strong>$1</strong>");
    safe=safe.split("\n").map(line=>/^#{1,4}\s/.test(line)?"<h4>"+line.replace(/^#{1,4}\s+/,"")+"</h4>":
         /^[-*]\s/.test(line)?'<div class="list-line">• '+line.slice(2)+"</div>":line).join("<br>");
    safe=safe.replace(/(<\/h4>)<br>/g,"$1").replace(/<br>(<h4>)/g,"$1");
    return safe.replace(/\u0000FRAGMENT(\d+)\u0000/g,(_,index)=>fragments[+index]||"");
  }
  function renderMath(root) {
    if(!window.katex)return;
    for(const element of root.querySelectorAll("[data-math-pending]")){
      const tex=element.getAttribute("data-tex");
      try {
        // Use DOM rendering; no model-provided HTML, external image commands or shared macros.
        window.katex.render(tex,element,{displayMode:element.dataset.display==="1",throwOnError:false,
          trust:false,strict:"ignore",maxExpand:500,maxSize:12,macros:{},output:"htmlAndMathml"});
        element.removeAttribute("data-math-pending");
        element.removeAttribute("data-tex");
        element.classList.remove("math-pending");
      }catch(error){
        element.textContent=tex;
        element.classList.add("math-source");
      }
    }
  }
  window.TinyLLMRender={markdown,renderMath};
})();
