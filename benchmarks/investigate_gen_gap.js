const R=require("./lib/runners");
const M=require("./lib/model");
(async()=>{
 const model=M.loadModel(); const w={kind:"generate",prompt:M.PROMPT,newTokens:10,expectedTokens:M.REFERENCE_GENERATION};
 R.runLocal(model,w,{warmup:3,iterations:0}); await R.runDistributed(model,w,{warmup:3,iterations:0,maxMessageSize:262144});
 const L=[],D=[];
 for(let r=0;r<8;r++){ const lf=()=>L.push(R.runLocal(model,w,{warmup:0,iterations:1}).rows[0].totalMs);
   const df=async()=>D.push((await R.runDistributed(model,w,{warmup:0,iterations:1,maxMessageSize:262144})).rows[0].totalMs);
   if(r%2===0){lf();await df();}else{await df();lf();} }
 const f=a=>a.map(x=>x.toFixed(0)).join(" ");
 console.log("local",f(L));console.log("dist ",f(D));
 const med=a=>a.slice().sort((x,y)=>x-y)[a.length>>1];console.log("median local",med(L).toFixed(0),"dist",med(D).toFixed(0));
})();
