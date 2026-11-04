'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (value, digits=2) => value == null ? '—' : Number(value).toFixed(digits);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const api = new URL('./api/v1/', location.href);
const state = {today:null, version:null, pickDate:null, detailCode:null, filter:null, filterPage:1, sort:'ret', descending:true, selected:new Map(), tokens:{}, chart:null};

async function request(path, options={}) {
  const response = await fetch(new URL(path, api), {cache:'no-store', ...options});
  let data;
  try { data = await response.json(); } catch { throw new Error('服务暂时不可用，请稍后重试。'); }
  if (!response.ok) {
    const message = Array.isArray(data.detail) ? data.detail.map(x => x.msg).join('；') : data.detail;
    throw new Error(message || '请求失败，请稍后重试。');
  }
  return data;
}
function notice(message, type='') { return `<div class="notice ${type}" role="status">${esc(message)}</div>`; }
function loading(target, message) { $(target).innerHTML = `<div class="loading" role="status"><span class="spinner"></span>${esc(message)}</div>`; }
function invalidate(target) { state.tokens[target] = Symbol(); }
async function task(args, target, render) {
  const token = Symbol(); state.tokens[target] = token;
  loading(target, '正在提交查询…');
  try {
    let job = await request('jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(args)});
    let retried=false;
    while (state.tokens[target] === token) {
      if (job.status === 'error') {
        if(!retried && /^服务已/.test(job.error||'')){
          retried=true;await sleep(250);
          if(state.tokens[target]!==token)return;
          job=await request('jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)});
          continue;
        }
        throw new Error(job.error || '计算失败，请重试。');
      }
      if (job.status === 'completed') {
        const result = await request(`jobs/${job.id}`);
        if (state.tokens[target] === token) render(result.result, job.id);
        return;
      }
      loading(target, job.status === 'queued' ? '查询已排队，稍后自动显示结果…' : '正在计算，完成后自动显示结果…');
      await sleep(1200);
      if (state.tokens[target] !== token) return;
      job = await request(`jobs/${job.id}`);
    }
  } catch (error) {
    if (state.tokens[target] === token) $(target).innerHTML = notice(error.message, 'error');
  }
}
function switchPane(name) {
  document.querySelectorAll('[data-pane]').forEach(button => {
    button.classList.toggle('active', button.dataset.pane === name);
    button.setAttribute('aria-selected', String(button.dataset.pane === name));
  });
  for (const pane of ['pick','filter','detail']) $(`pane-${pane}`).hidden = pane !== name;
  history.replaceState(null, '', `#${name}`);
  if (name === 'detail' && state.chart) requestAnimationFrame(() => drawChart(state.chart.range));
}
document.querySelectorAll('[data-pane]').forEach(button => button.addEventListener('click', () => switchPane(button.dataset.pane)));

function codeLink(code) { return `<a href="#detail" data-code="${esc(code)}">${esc(code)}</a>`; }
document.addEventListener('click', event => {
  const link = event.target.closest('[data-code]');
  if (!link) return;
  event.preventDefault(); switchPane('detail'); $('detail-code').value = link.dataset.code;
  $('detail-form').requestSubmit();
});
function table(rows, columns, options={}) {
  if (!rows.length) return '<div class="empty">暂无数据。</div>';
  const head = columns.map(c => `<th class="${c.text?'text':''}">${options.sortable ? `<button data-sort="${c.key}">${esc(c.label)}${state.sort===c.key?(state.descending?' ↓':' ↑'):''}</button>` : esc(c.label)}</th>`).join('');
  const body = rows.map(row => `<tr class="${row.role==='★ 选中'?'chosen':''} ${options.shared?.has(String(row['代码']))?'shared':''}">${options.selectable?`<td><input type="checkbox" data-select="${esc(row.code)}" aria-label="选择 ${esc(row.name)}" ${state.selected.has(row.code)?'checked':''}></td>`:''}${columns.map(c => `<td class="${c.text?'text':''}">${c.render ? c.render(row[c.key],row) : esc(row[c.key]??'—')}</td>`).join('')}</tr>`).join('');
  return `<div class="table-wrap"><table><thead><tr>${options.selectable?'<th>选择</th>':''}${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
const pickColumns = [
  {key:'rank',label:'跌幅榜',render:v=>v??'—'}, {key:'code',label:'基金代码',render:codeLink},
  {key:'name',label:'基金名称',text:true}, {key:'ret3m',label:'近3月收益(%)',render:v=>num(v)},
  {key:'win_start',label:'近3月窗口',render:(v,r)=>v&&r.win_end?`${esc(v)} → ${esc(r.win_end)}`:'—'},
  {key:'beta',label:'波动率比值',render:v=>num(v)}, {key:'aum',label:'规模(亿)',render:v=>num(v)},
  {key:'why',label:'结论',text:true,render:(v,r)=>esc(v||r.role||'通过')}
];
function renderPick(result) {
  let html = notice(`数据够：本次需要截至 ${result.need_date} 的净值，净值库已到 ${result.nav_max}。`);
  const s = result.std;
  html += `<p class="muted">近3月跌幅最大 · 跌幅 ≤ ${num(s.max_drop,0)}% · 波动率比值 ≥ ${s.min_vol_ratio} · 规模 ≥ ${s.min_aum} 亿${s.max_aum?'，上限 '+s.max_aum+' 亿':''}。${result.too_deep} 只跌幅超过上限。</p>`;
  if (result.asof !== state.today && result.thr != null) {
    html += notice(result.broken ? `${result.qdate} 期权链失效，按触发信号处理，与回测一致。` :
      `${result.qdate} QVIX ${num(result.qvix)}，阈值 ${num(result.thr)}：${result.hit?'触发买入信号。':'没有信号，下面仅展示假如买会选哪只。'}`, result.hit?'':'warn');
  }
  if (result.thr == null) html += notice('这一天没有足够的 QVIX 阈值历史，请换个日期。','warn');
  else html += table(result.rows, pickColumns);
  if (result.chosen) html += notice(`★ ${result.chosen_name||result.chosen}（${result.chosen}）　基金回撤控制线 ${num(result.fund_line)}%　大盘回撤线 ${num(result.sse_line)}%`);
  else if (result.thr != null) html += notice('没有符合全部门槛的候选，按策略放弃当天。','warn');
  $('pick-output').innerHTML = html;
  $('probe-panel').hidden = result.thr == null;
  state.pickDate = result.asof;
}
$('pick-form').addEventListener('submit', event => {
  event.preventDefault(); $('probe-panel').hidden=true; invalidate('probe-output'); $('probe-output').innerHTML='';
  task({operation:'pick',asof:$('pick-date').value}, 'pick-output', renderPick);
});
$('pick-date').addEventListener('change', () => {
  invalidate('pick-output'); invalidate('probe-output'); state.pickDate=null; $('probe-panel').hidden=true;
  $('pick-output').innerHTML=notice('日期已改变，请重新计算。');
});
$('probe-form').addEventListener('submit', event => {
  event.preventDefault();
  const codes=$('probe-codes').value.trim().split(/[,，\s]+/).filter(Boolean);
  if (codes.length>20 || codes.some(c=>!/^\d{6}$/.test(c))) { $('probe-output').innerHTML=notice('请输入最多 20 个六位基金代码。','error'); return; }
  task({operation:'probe',asof:state.pickDate,codes},'probe-output',r=>{
    $('probe-output').innerHTML=table(r.rows,pickColumns)+`<p class="muted">同为 ${esc(r.asof)} 的数据，真跌基金共 ${r.n_drop??'—'} 只。</p>`;
  });
});

const filterColumns=[{key:'code',label:'基金代码',render:codeLink},{key:'name',label:'基金名称',text:true},{key:'type',label:'类型',text:true},{key:'aum',label:'规模(亿)',render:v=>num(v)},{key:'ret',label:'收益率(%)',render:v=>num(v)},{key:'mdd',label:'最大回撤(%)',render:v=>num(v)}];
function renderFilter(result,id) {
  if (!state.filter || state.filter.id!==id) state.selected.clear();
  state.filter={id,result}; state.filterPage=result.page||1;
  const period={'1m':'近1月','3m':'近3月','6m':'近6月','1y':'近1年'}[result.period];
  const columns=filterColumns.map(c=>({...c,label:c.key==='ret'?`${period}收益率(%)`:c.key==='mdd'?`${period}回撤(%)`:c.label}));
  $('filter-output').innerHTML=`<p class="muted">共 ${result.total} 条匹配 · ${result.historical?'历史快照：只使用 '+esc(result.asof)+' 前的净值':'最新预计算指标'} · 净值截至 ${esc(result.freshness.nav_date)}</p>`+table(result.rows,columns,{selectable:true,sortable:true});
  $('filter-controls').hidden=false; const pages=Math.max(1,Math.ceil(result.total/Number($('filter-size').value)));
  $('filter-page').textContent=`${state.filterPage} / ${pages}`;
  $('filter-prev').disabled=state.filterPage<=1; $('filter-next').disabled=state.filterPage>=pages;
  $('filter-download').href=new URL(`jobs/${id}/download`,api);
  updateSelected();
}
const optionalNumber=id=>$(id).value.trim()===''?null:Number($(id).value);
$('filter-form').addEventListener('submit',event=>{
  event.preventDefault(); $('filter-controls').hidden=true; $('compare-output').innerHTML=''; invalidate('compare-output');
  state.selected.clear(); state.sort='ret';state.descending=true;
  task({operation:'filter',period:$('filter-period').value,types:[...$('filter-types').selectedOptions].map(x=>x.value),min_ret:optionalNumber('filter-ret'),max_dd:optionalNumber('filter-dd'),min_aum:Number($('filter-aum').value)||0,asof:$('filter-date').value||null},'filter-output',renderFilter);
});
async function filterPage(page) {
  if (!state.filter) return;
  const id=state.filter.id; const token=Symbol();state.tokens['filter-output']=token;
  try{
    const r=await request(`jobs/${id}?page=${page}&page_size=${$('filter-size').value}&sort=${state.sort}&descending=${state.descending}`);
    if(state.tokens['filter-output']===token)renderFilter(r.result,id);
  }catch(e){if(state.tokens['filter-output']===token)$('filter-output').innerHTML=notice(e.message,'error');}
}
$('filter-prev').onclick=()=>filterPage(state.filterPage-1);$('filter-next').onclick=()=>filterPage(state.filterPage+1);$('filter-size').onchange=()=>filterPage(1);
$('filter-output').addEventListener('click',event=>{
  const sort=event.target.closest('[data-sort]');if(!sort)return;
  state.descending=state.sort===sort.dataset.sort?!state.descending:true;state.sort=sort.dataset.sort;filterPage(1);
});
function updateSelected(){const n=state.selected.size;$('compare-button').disabled=!n;$('compare-button').textContent=`查看已选持仓${n?'（'+n+'）':''}`;}
$('filter-output').addEventListener('change',event=>{
  const code=event.target.dataset.select;if(!code)return;
  if(event.target.checked){if(state.selected.size>=5){event.target.checked=false;return;}state.selected.set(code,state.filter.result.rows.find(r=>r.code===code));}else state.selected.delete(code);updateSelected();
});
async function resultTask(args){
  let job=await request('jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)});
  while(job.status!=='completed'){if(job.status==='error')throw new Error(job.error);await sleep(1200);job=await request(`jobs/${job.id}`);}
  return (await request(`jobs/${job.id}`)).result;
}
$('compare-button').onclick=async()=>{
  const token=Symbol();state.tokens['compare-output']=token;const selected=[...state.selected.values()];
  loading('compare-output','正在逐只加载已选基金的持仓…');
  try{
    const entries=[];const asof=state.filter.result.asof;const ends={'1':'-03-31','2':'-06-30','3':'-09-30','4':'-12-31'};
    for(const fund of selected){
      if(state.tokens['compare-output']!==token)return;
      const result=await resultTask({operation:'holdings',code:fund.code,start_year:Math.max(2020,Number(asof.slice(0,4))-1)});
      const rows=result.rows.filter(r=>r.quarter.slice(0,4)+ends[r.quarter.slice(-1)]<=asof);
      const quarter=rows.length?rows[0].quarter:null;entries.push({fund,quarter,source:result.source,rows:rows.filter(r=>r.quarter===quarter)});
    }
    const hits=new Map();for(const e of entries){for(const c of new Set(e.rows.map(r=>r['代码'])))hits.set(c,(hits.get(c)||0)+1);}
    const shared=new Set([...hits].filter(([,n])=>n>=2).map(([c])=>c));
    if(state.tokens['compare-output']!==token)return;
    $('compare-output').innerHTML=`<h3>已选基金最新十大持仓</h3><p class="muted">截至 ${esc(asof)}，按季度末选择最新一期；相同底色表示共同持仓。季度披露通常滞后约两周。</p><div class="hold-cards">${entries.map(e=>`<div class="hold-card"><h3>${esc(e.fund.code)} ${esc(e.fund.name)}</h3><p class="muted">${esc(e.quarter||'暂无季度数据')}${e.source?' · 穿透至 '+esc(e.source.join(' ')):''}</p>${table(e.rows,holdingColumns,{shared})}</div>`).join('')}</div>`;
  }catch(e){if(state.tokens['compare-output']===token)$('compare-output').innerHTML=notice(e.message,'error');}
};

function kpi(label,value){return `<div class="kpi"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;}
function renderDetail(result){
  const r=result.info,m=result.metrics||{};state.detailCode=r.code;
  let html=`<h3>${esc(r.name)}（${esc(r.code)}）</h3><p class="muted">${esc(r.type||'')} · ${result.source==='eastmoney'?'净值来自本次东财查询':'净值来自已发布快照'}</p>`;
  const last=result.nav.at(-1);
  html+='<div class="kpis">'+kpi('单位净值',num(last.nav,4))+kpi('净值日期',last.date)+kpi('年化收益',m.ann_return==null?'—':num(m.ann_return*100)+'%')+kpi('年化波动率',m.volatility==null?'—':num(m.volatility*100)+'%')+'</div>';
  html+='<div class="chart-card"><div class="toolbar" id="chart-ranges">'+[['1m','近1月'],['3m','近3月'],['6m','近6月'],['1y','近1年'],['all','全部']].map(([key,label])=>`<button type="button" data-range="${key}">${label}</button>`).join('')+'</div><div id="chart-tip" class="chart-tip"></div><svg class="chart" id="nav-chart" role="img" aria-label="单位净值走势图"></svg></div>';
  html+='<h3>季度规模</h3>'+table(result.scales,[{key:'quarter_end',label:'报告日期'},{key:'aum',label:'期末净资产(亿)',render:v=>num(v)},{key:'publish_date',label:'估算披露日期'}]);
  html+='<details><summary>净值历史（'+result.nav.length+' 个数据点）</summary>'+table([...result.nav].reverse(),[{key:'date',label:'净值日期'},{key:'nav',label:'单位净值',render:v=>num(v,4)},{key:'acc_nav',label:'累计净值',render:v=>num(v,4)},{key:'daily_ret_pct',label:'日增长(%)',render:v=>num(v)}])+'</details>';
  $('detail-output').innerHTML=html;$('holdings-panel').hidden=false;$('holdings-output').innerHTML='';$('holdings-history-button').hidden=true;
  state.chart={rows:result.nav,range:'1y'};
  $('chart-ranges').onclick=e=>{const button=e.target.closest('[data-range]');if(button)drawChart(button.dataset.range);};
  drawChart('1y');
}
$('detail-form').addEventListener('submit',event=>{
  event.preventDefault();state.detailCode=null;state.chart=null;$('holdings-panel').hidden=true;invalidate('holdings-output');
  task({operation:'detail',code:$('detail-code').value},'detail-output',renderDetail);
});
function drawChart(range){
  if(!state.chart||!$('nav-chart'))return;state.chart.range=range;
  const source=state.chart.rows;const days={'1m':30,'3m':91,'6m':182,'1y':365};
  const cutoff=Date.parse(source.at(-1).date)-(days[range]||100000)*86400000;
  const rows=source.filter(r=>Date.parse(r.date)>=cutoff&&Number.isFinite(r.nav));if(!rows.length)return;
  const svg=$('nav-chart'),w=Math.max(svg.clientWidth,300),h=svg.clientHeight||300,L=52,R=12,T=16,B=30;
  const vals=rows.map(r=>r.nav),low=Math.min(...vals),high=Math.max(...vals),pad=Math.max((high-low)*.08,.001),min=low-pad,max=high+pad;
  const t0=Date.parse(rows[0].date),span=Math.max(Date.parse(rows.at(-1).date)-t0,1);
  const x=r=>L+(Date.parse(r.date)-t0)/span*(w-L-R),y=v=>T+(max-v)/(max-min)*(h-T-B);
  let html='';for(let i=0;i<5;i++){const v=min+(max-min)*i/4;html+=`<line class="grid" x1="${L}" x2="${w-R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end">${num(v,3)}</text>`;}
  html+=`<path class="trace" d="${rows.map((r,i)=>(i?'L':'M')+x(r).toFixed(1)+','+y(r.nav).toFixed(1)).join(' ')}"/><line class="cursor" id="chart-cursor" y1="${T}" y2="${h-B}"/><text x="${L}" y="${h-5}">${esc(rows[0].date)}</text><text x="${w-R}" y="${h-5}" text-anchor="end">${esc(rows.at(-1).date)}</text>`;
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.innerHTML=html;
  $('chart-tip').textContent=`${rows.at(-1).date}　单位净值 ${num(rows.at(-1).nav,4)}`;
  svg.onpointermove=e=>{const pos=e.clientX-svg.getBoundingClientRect().left;const target=t0+(pos-L)/(w-L-R)*span;
    let lo=0,hi=rows.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(Date.parse(rows[mid].date)<target)lo=mid+1;else hi=mid;}
    const row=rows[lo];const cursor=$('chart-cursor');cursor.style.display='block';cursor.setAttribute('x1',x(row));cursor.setAttribute('x2',x(row));$('chart-tip').textContent=`${row.date}　单位净值 ${num(row.nav,4)}　日增长 ${num(row.daily_ret_pct)}%`;
  };svg.onpointerleave=()=>{if($('chart-cursor'))$('chart-cursor').style.display='none';};
  $('chart-ranges').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.range===range));
}
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>state.chart&&drawChart(state.chart.range),100);});
const holdingColumns=[{key:'kind',label:'类型',text:true},{key:'代码',label:'代码'},{key:'名称',label:'名称',text:true},{key:'占净值比例',label:'占净值比例(%)',render:v=>num(v)},{key:'持股数',label:'持股数(万股)',render:v=>num(v)},{key:'持仓市值',label:'市值(万元)',render:v=>num(v)}];
function loadHoldings(startYear){
  if(!state.detailCode)return;
  task({operation:'holdings',code:state.detailCode,start_year:startYear},'holdings-output',result=>{
    $('holdings-history-button').hidden=startYear===2020;
    const quarters=[...new Set(result.rows.map(r=>r.quarter))];
    if(!quarters.length){$('holdings-output').innerHTML=notice('该基金暂无披露的重仓持仓。');return;}
    $('holdings-output').innerHTML=(result.source?notice('联接基金，重仓穿透至 '+result.source.join(' ')):'')+'<div id="hold-quarters" class="quarter-tabs">'+quarters.map(q=>`<button type="button" data-quarter="${esc(q)}">${esc(q)}</button>`).join('')+'</div><div id="quarter-table"></div>';
    const show=q=>{$('quarter-table').innerHTML=table(result.rows.filter(r=>r.quarter===q),holdingColumns);$('hold-quarters').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.quarter===q));};
    $('hold-quarters').onclick=e=>{const b=e.target.closest('[data-quarter]');if(b)show(b.dataset.quarter);};show(quarters[0]);
  });
}
$('holdings-button').onclick=()=>loadHoldings(Number((state.today||new Date().toISOString()).slice(0,4)));
$('holdings-history-button').onclick=()=>loadHoldings(2020);

async function refreshStatus(){
  try{
    const r=await request('status');state.today=r.today;
    if(state.version&&state.version!==r.version){
      for(const target of ['pick-output','probe-output','filter-output','compare-output','detail-output','holdings-output']){
        invalidate(target);$(target).innerHTML='';
      }
      for(const target of ['pick-output','filter-output','detail-output'])$(target).innerHTML=notice('数据已更新，请重新查询。');
      $('probe-panel').hidden=true;$('filter-controls').hidden=true;$('holdings-panel').hidden=true;
      state.pickDate=null;state.detailCode=null;state.chart=null;state.filter=null;state.selected.clear();
    }
    state.version=r.version;
    if(!$('pick-date').value)$('pick-date').value=r.today;
    $('pick-date').max=r.today;$('filter-date').max=r.today;
    const sync=new Date(r.synced_at*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});
    $('data-status').textContent=`净值 ${r.nav_date} · 上证 ${r.sse_date} · QVIX ${r.qvix_date||'—'} · ${r.nav_funds} 只基金有净值 · 同步于 ${sync}`;
    if(!$('filter-types').options.length)$('filter-types').innerHTML=r.types.map(t=>`<option value="${esc(t)}">${esc(t)}</option>`).join('');
  }catch(e){$('data-status').innerHTML=notice(e.message,'warn');}
}
switchPane(['pick','filter','detail'].includes(location.hash.slice(1))?location.hash.slice(1):'pick');
refreshStatus();setInterval(refreshStatus,60000);
