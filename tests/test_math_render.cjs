const {JSDOM}=require("./dom/node_modules/jsdom"),fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
const path=require("path"),root=path.resolve(__dirname,".."),dom=new JSDOM("<!doctype html><div id='out'></div>",{runScripts:"outside-only"}),w=dom.window;
vm.runInContext(fs.readFileSync(root+"/ui/vendor/katex/katex.min.js","utf8"),dom.getInternalVMContext());
vm.runInContext(fs.readFileSync(root+"/ui/rendering.js","utf8"),dom.getInternalVMContext());
const out=w.document.getElementById("out"),results=[];
function render(source){out.innerHTML=w.TinyLLMRender.markdown(source);w.TinyLLMRender.renderMath(out);}
function check(name,fn){fn();results.push({name,passed:true});console.log(name,"PASS");}
check("inline delimiters",()=>{render("Solve \\(x^2+1=y\\), isolate \\(x^2\\).");assert.equal(out.querySelectorAll(".katex").length,2);assert.ok(out.querySelector("msup"));});
check("double escaped delimiters and command",()=>{render("Solve \\\\(x^2+1=y\\\\). \\\\[x=\\\\frac{-b\\\\pm\\\\sqrt{b^2-4ac}}{2a}\\\\]");assert.equal(out.querySelectorAll(".katex").length,2);assert.ok(out.querySelector(".mfrac"));});
check("multiline display math",()=>{render("A:\n\\[\nx=\\pm\\sqrt{y-1}\n\\]\nB.");assert.equal(out.querySelectorAll(".katex-display").length,1);assert.equal(out.querySelectorAll(".katex-error").length,0);});
check("dollar delimiters",()=>{render("$x^2$ and $$\\frac{1}{2}$$");assert.equal(out.querySelectorAll(".katex").length,2);});
check("nested standalone boxed",()=>{render("\\boxed{\\frac{1}{2}}");assert.ok(out.querySelector(".mfrac"));});
check("matrix row break retained",()=>{render("\\[\\begin{matrix}1&2\\\\3&4\\end{matrix}\\]");assert.ok(out.querySelector(".mtable"));assert.equal(out.querySelectorAll(".katex-error").length,0);});
check("math inside code stays literal",()=>{render("\x60\x60\x60latex\n\\(x^2\\)\n\x60\x60\x60\n\x60$a$\x60");assert.equal(out.querySelectorAll(".katex").length,0);assert.ok(out.querySelector("pre code").textContent.includes("\\("));});
check("partial streaming formula stays visible",()=>{render("Compute \\(x^2+");assert.equal(out.querySelectorAll(".katex").length,0);assert.ok(out.textContent.includes("x^2+"));render("Compute \\(x^2+1=y\\)");assert.equal(out.querySelectorAll(".katex").length,1);});
check("untrusted commands cannot load images",()=>{render("\\(\\includegraphics{https://example.com/leak}\\)");assert.equal(out.querySelectorAll("img,script,iframe").length,0);});
check("model HTML escaped",()=>{render('<img src=x onerror="alert(1)">');assert.equal(out.querySelectorAll("img").length,0);});
check("representative equation renders",()=>{render("Given \\(x^2+1=y\\), then \\(x=\\pm\\sqrt{y-1}\\). Also \\(a^2+b^2=c^2\\).");assert.ok(out.querySelectorAll(".katex").length>=3);});
dom.window.close();
