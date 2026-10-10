const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function element(){return {innerHTML:'',textContent:'',value:'question',checked:true,disabled:false,dataset:{},addEventListener(){},querySelectorAll(){return []},classList:{add(){},remove(){}}};}
const document={querySelectorAll(){return []},getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},createElement(){let text='';return {set textContent(v){text=String(v)},get innerHTML(){return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}}}};
const context=vm.createContext({document,console,setTimeout,fetch:async()=>({ok:true,json:async()=>({documents:[],config:{generation_provider:'groq',generation_model:'test-model'}})})});
vm.runInContext(fs.readFileSync(require('path').join(__dirname,'../static/app.js'),'utf8'),context);
let checks=0;function check(name,fn){fn();checks++;console.log('PASS '+name)}
(async()=>{
 await context.loadDocuments();
 check('cloud notice',()=>assert.match(elements.get('generation-notice').textContent,/passages to Groq/));
 context.fetch=async()=>({ok:true,json:async()=>({documents:[],config:{}})});
 await context.loadDocuments();
 check('older server not mislabeled local',()=>assert.match(elements.get('generation-notice').textContent,/settings unavailable/));
 check('score scale',()=>assert.equal(context.formatEvidence(.362985),'0.363'));
 check('negative cosine',()=>assert.equal(context.formatEvidence(-.5),'-0.500'));
 check('invalid evidence',()=>{for(const v of [null,undefined,NaN,Infinity,'0.8',1.2])assert.equal(context.formatEvidence(v),'Unavailable')});
 check('HTML and quotes escaped',()=>assert.equal(context.escapeHtml('<x "a" \'b\'>'),'&lt;x &quot;a&quot; &#39;b&#39;&gt;'));
 context.renderAnswer('question',{status:'blocked',confidence:0,retrieval_score:.379,answer:'Insufficient evidence'});
 check('blocked evidence distinct from confidence',()=>assert.match(elements.get('answer-container').innerHTML,/Evidence: 0.379/));
 check('probability warning',()=>assert.match(elements.get('answer-container').innerHTML,/not a correctness probability/));
 context.renderAnswer('question',{status:'success',confidence:.8,answer:'safe',citations:['<script>x</script>'],query_rewritten:'<img src=x onerror=x>'});
 check('metadata cannot inject HTML',()=>assert(!elements.get('answer-container').innerHTML.includes('<script>')&&!elements.get('answer-container').innerHTML.includes('<img')));
 context.renderAnswer('question',{answer:'missing status'});
 check('unknown response not success',()=>assert(!elements.get('answer-container').innerHTML.includes('Answer Generated')));
 context.fetch=async()=>({ok:false,json:async()=>({detail:'failure'})});
 await context.submitQuery();
 check('HTTP failure not generated answer',()=>assert.match(elements.get('answer-container').innerHTML,/Query request failed/));
 check('button restored on HTTP failure',()=>assert.equal(elements.get('submit-btn').disabled,false));
 console.log(checks+' frontend checks passed; zero live requests');
})().catch(e=>{console.error(e);process.exitCode=1});
