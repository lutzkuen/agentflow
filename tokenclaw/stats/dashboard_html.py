from __future__ import annotations


def dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TokenClaw Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#0f1419;color:#d8dee9;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px}
  header{align-items:center;background:#151b22;border-bottom:1px solid #2c3642;display:flex;gap:12px;padding:16px 24px}
  h1{color:#f3f6fa;font-size:18px;font-weight:650;letter-spacing:0;margin:0}
  .sub{color:#93a1b1;font-size:12px}
  #status{color:#93a1b1;font-size:12px;margin-left:auto}
  .tabs{border-bottom:1px solid #2c3642;display:flex;gap:4px;padding:0 24px}
  .tab-btn{background:transparent;border:0;border-bottom:2px solid transparent;color:#93a1b1;cursor:pointer;font:inherit;padding:12px 14px}
  .tab-btn.active{border-bottom-color:#5aa7ff;color:#f3f6fa}
  .tab-panel{display:none;padding:20px 24px 28px}
  .tab-panel.active{display:block}
  .cards{display:grid;gap:12px;grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:18px}
  .card{background:#151b22;border:1px solid #2c3642;border-radius:8px;min-width:0;padding:14px}
  .label{color:#93a1b1;font-size:11px;font-weight:650;text-transform:uppercase}
  .value{color:#f3f6fa;font-size:23px;font-weight:700;margin-top:6px;overflow-wrap:anywhere}
  .hint{color:#93a1b1;font-size:12px;line-height:1.4;margin-top:4px;overflow-wrap:anywhere}
  .money{color:#6ad77f}
  .warn{color:#ffbd5e}
  .bad{color:#ff7b72}
  .section{margin-top:18px}
  h2{color:#93a1b1;font-size:12px;font-weight:650;letter-spacing:.04em;margin:0 0 10px;text-transform:uppercase}
  .table-wrap{overflow-x:auto}
  table{border-collapse:collapse;min-width:900px;width:100%}
  th{border-bottom:1px solid #2c3642;color:#93a1b1;font-size:11px;font-weight:650;padding:8px 10px;text-align:left;text-transform:uppercase;white-space:nowrap}
  td{border-bottom:1px solid #1d252e;padding:8px 10px;white-space:nowrap}
  tr:hover td{background:#151b22}
  .badge{border-radius:4px;display:inline-block;font-size:11px;font-weight:650;padding:2px 6px}
  .badge.ok{background:#17351f;color:#6ad77f}
  .badge.err{background:#391b1b;color:#ff7b72}
  .badge.miss{background:#252c34;color:#b7c0cc}
  .model{max-width:260px;overflow:hidden;text-overflow:ellipsis}
  .muted{color:#93a1b1}
  @media (max-width:900px){
    header{padding:14px 16px}
    .tabs{padding:0 16px}
    .tab-panel{padding:16px}
    .cards{grid-template-columns:repeat(2,minmax(0,1fr))}
  }
  @media (max-width:560px){
    header{align-items:flex-start;flex-direction:column}
    #status{margin-left:0}
    .cards{grid-template-columns:1fr}
  }
</style>
</head>
<body>
<header>
  <h1>TokenClaw</h1>
  <span class="sub">local savings dashboard</span>
  <span id="status">loading</span>
</header>

<nav class="tabs" aria-label="Dashboard views">
  <button class="tab-btn active" type="button" data-tab-name="today" onclick="showTab('today')">Today</button>
  <button class="tab-btn" type="button" data-tab-name="last7" onclick="showTab('last7')">Last 7 days</button>
</nav>

<main>
  <section class="tab-panel active" id="tab-today">
    <div class="cards">
      <div class="card"><div class="label">Calls today</div><div class="value" id="today-calls">-</div><div class="hint" id="today-errors">- errors</div></div>
      <div class="card"><div class="label">Spend today</div><div class="value money" id="today-spend">-</div><div class="hint" id="today-tokens">- tokens</div></div>
      <div class="card"><div class="label">Saved today</div><div class="value money" id="today-savings">-</div><div class="hint" id="today-savings-detail">local routing, crunching, exact cache</div></div>
      <div class="card"><div class="label">Health</div><div class="value" id="today-health">-</div><div class="hint" id="today-latency">- avg latency</div></div>
    </div>
    <div class="section">
      <h2>Recent Calls</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Time</th><th>Provider</th><th>Category</th><th>Requested</th><th>Routed</th><th>Spend</th><th>Saved</th><th>Status</th><th>Latency</th>
          </tr></thead>
          <tbody id="recent-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="tab-panel" id="tab-last7">
    <div class="cards">
      <div class="card"><div class="label">Units</div><div class="value" id="week-units">-</div><div class="hint" id="week-success">- successful</div></div>
      <div class="card"><div class="label">Spend</div><div class="value money" id="week-spend">-</div><div class="hint" id="week-baseline">- baseline</div></div>
      <div class="card"><div class="label">Savings</div><div class="value money" id="week-savings">-</div><div class="hint">local summary endpoint</div></div>
      <div class="card"><div class="label">Tokens</div><div class="value" id="week-tokens">-</div><div class="hint" id="week-errors">- errors</div></div>
      <div class="card"><div class="label">Managed feed</div><div class="value" id="week-managed-state">-</div><div class="hint" id="week-managed-detail">-</div></div>
      <div class="card"><div class="label">Routing conversion</div><div class="value" id="week-routing-conversion">-</div><div class="hint" id="week-routing-conversion-detail">-</div></div>
    </div>
    <div class="section">
      <h2>Managed Backing</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Window</th><th>Mode</th><th>Server</th><th>Auth</th><th>Attempted</th><th>Succeeded</th><th>Skipped</th><th>Manual</th><th>Managed</th><th>Off / pass-through</th>
          </tr></thead>
          <tbody id="week-managed-tbody"></tbody>
        </table>
      </div>
    </div>
    <div class="section">
      <h2>Daily Breakdown</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Date</th><th>Units</th><th>Provider calls</th><th>Codex turns</th><th>Success</th><th>Errors</th><th>Tokens</th><th>Spend</th><th>Baseline</th><th>Savings</th>
          </tr></thead>
          <tbody id="weekly-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>
</main>

<script>
const statusEl=document.getElementById('status');
function text(id,value){document.getElementById(id).textContent=value}
function money(value){return '$'+Number(value||0).toFixed(4)}
function num(value){return Number(value||0).toLocaleString()}
function esc(value){
  return String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function fmtSec(value){
  if(value==null)return '-';
  const n=Number(value||0);
  if(n<60)return `${Math.round(n)}s`;
  if(n<3600)return `${Math.round(n/60)}m`;
  if(n<86400)return `${Math.round(n/3600)}h`;
  return `${Math.round(n/86400)}d`;
}
function localTime(value){
  if(!value)return '-';
  const d=new Date(value);
  return Number.isNaN(d.getTime())?'-':d.toLocaleString();
}
function statusBadge(code){
  const n=Number(code||0);
  if(n>=400)return `<span class="badge err">${n}</span>`;
  if(n>=200)return `<span class="badge ok">${n}</span>`;
  return '<span class="badge miss">-</span>';
}
function managedSummaryRow(label,state,summary){
  state=state||{};
  summary=summary||{};
  const calls=summary.policy_decision_calls||{};
  const backing=summary.backing_counts||{};
  const managed=Number(backing['managed-recommended']||0)+Number(backing['managed-enforced']||0);
  const server=(state.server||{}).configured?'configured':'off';
  const auth=(state.server||{}).auth_configured?'configured':'not configured';
  return `<tr>
    <td>${label}</td>
    <td>${state.mode||'local_only'}</td>
    <td>${server}</td>
    <td>${auth}</td>
    <td>${num(calls.attempted)}</td>
    <td>${num(calls.succeeded)}</td>
    <td>${num(calls.skipped)}</td>
    <td>${num(backing['local-manual'])}</td>
    <td>${num(managed)}</td>
    <td>${num(backing['off/pass-through'])}</td>
  </tr>`;
}
function managedCompactText(summary){
  summary=summary||{};
  const calls=summary.policy_decision_calls||{};
  const backing=summary.backing_counts||{};
  const managed=Number(backing['managed-recommended']||0)+Number(backing['managed-enforced']||0);
  return `${num(calls.succeeded)} succeeded · ${num(calls.skipped)} skipped · ${num(managed)} managed-backed`;
}
function renderRoutingConversion(conv){
  conv=conv||{};
  const recommended=Number(conv.route_recommended||0);
  const applied=Number(conv.applied||0);
  const value=document.getElementById('week-routing-conversion');
  const detail=document.getElementById('week-routing-conversion-detail');
  if(!value||!detail)return;
  value.textContent=`${num(applied)}/${num(recommended)} applied`;
  if(!recommended){
    value.className='value muted';
    detail.textContent='no routes recommended';
    return;
  }
  const held=Number(conv.held||0);
  const rate=conv.applied_rate==null?null:Math.round(Number(conv.applied_rate)*100);
  // A recommended-but-never-applied funnel is the exact bottleneck this surfaces.
  value.className='value '+(applied?'money':'err');
  detail.textContent=held
    ? `${rate==null?'-':rate+'%'} applied · top hold: ${esc(conv.top_hold_reason||'unknown')}`
    : 'all recommended routes applied';
}
function queueStatusClass(status){
  if(status==='sent')return'ok';
  if(status==='retryable-error'||status==='dropped-after-limit'||status==='error')return'err';
  return'miss';
}
function expectationClass(state){
  if(state==='actionable')return'err';
  if(state==='watch')return'warn';
  if(state==='expected')return'miss';
  return'ok';
}
function showTab(name){
  document.querySelectorAll('.tab-btn').forEach(btn=>btn.classList.toggle('active',btn.dataset.tabName===name));
  document.querySelectorAll('.tab-panel').forEach(panel=>panel.classList.toggle('active',panel.id==='tab-'+name));
}
async function getJson(url){
  const response=await fetch(url,{cache:'no-store'});
  if(!response.ok)throw new Error(`${url} returned ${response.status}`);
  return response.json();
}
function renderToday(data){
  const summary=data.summary||{};
  const health=(data.executive_summary||{}).health||{};
  const savings=(data.executive_summary||{}).savings||{};
  const tokens=(data.executive_summary||{}).tokens_today||{};
  text('today-calls',num(data.today_calls));
  text('today-errors',`${num(summary.today_errors)} errors`);
  text('today-spend',money(data.today_cost_usd));
  text('today-tokens',`${num(tokens.total_tokens)} tokens`);
  text('today-savings',money(savings.today_tokenclaw_generated_savings_usd??data.today_savings_usd));
  const buckets=savings.today_tokenclaw_generated_buckets||{};
  const providerPromptCache=savings.today_provider_prompt_cache_discount_usd??0;
  text('today-savings-detail',`routing ${money(buckets.routing_usd)} · crunch ${money(buckets.crunching_usd)} · local cache ${money(buckets.exact_local_cache_usd)} · provider prompt-cache ${money(providerPromptCache)} separate`);
  text('today-health',summary.today_errors?'Check errors':'OK');
  document.getElementById('today-health').className='value '+(summary.today_errors?'warn':'money');
  text('today-latency',`${num(health.avg_latency_ms||summary.avg_latency_ms)} ms avg latency`);
  const rows=(data.recent||[]).map(row=>{
    const routed=row.routed_model&&row.routed_model!==row.requested_model?row.routed_model:'-';
    return `<tr>
      <td class="muted">${localTime(row.created_at)}</td>
      <td>${row.provider||'unknown'}</td>
      <td>${row.category||'unknown'}</td>
      <td class="model">${row.requested_model||'-'}</td>
      <td class="model">${routed}</td>
      <td class="money">${money(row.cost_est_usd)}</td>
      <td class="money">${money(row.saved_usd)}</td>
      <td>${statusBadge(row.status_code)}</td>
      <td class="muted">${num(row.latency_ms)} ms</td>
    </tr>`;
  }).join('');
  document.getElementById('recent-tbody').innerHTML=rows||'<tr><td colspan="9" class="muted">No calls recorded yet.</td></tr>';
}
function renderFeedbackFreshness(data){
  const tbody=document.getElementById('managed-feedback-freshness-tbody');
  if(!tbody)return;
  const groups=(data.groups||[]).slice(0,20);
  const rows=groups.map(row=>{
    const error=row.last_error_class||row.last_status_code?`HTTP ${row.last_status_code||''} ${row.last_error_class||''}`.trim():'none';
    return `<tr>
      <td><span class="badge miss">${esc(row.action_family||'unknown')}</span></td>
      <td>${esc(row.source_surface||'unknown')}</td>
      <td><span class="badge ${queueStatusClass(row.status)}">${esc(row.status||'unknown')}</span></td>
      <td>${num(row.row_count)}</td>
      <td>${num(row.due_count)}</td>
      <td>${num(row.attempt_count)}</td>
      <td>${fmtSec(row.oldest_queued_age_seconds)}</td>
      <td>${row.newest_queued_at?localTime(row.newest_queued_at):'-'}</td>
      <td><span class="badge ${expectationClass(row.expectation_state)}">${esc(row.expectation_state||'unknown')}</span></td>
      <td>${esc(error)}</td>
      <td>${row.payload_json_included?'<span class="badge err">payload included</span>':'<span class="badge ok">payload omitted</span>'}</td>
    </tr>`;
  }).join('');
  tbody.innerHTML=rows||'<tr><td colspan="11" class="muted">No managed feedback queue rows.</td></tr>';
}
function renderManagedActivation(data){
  data=data||{};
  const proof=data.activation_proof||{};
  const readiness=proof.thinking_tail_readiness||{};
  const feedback=data.feedback_burndown||{};
  const fresh=data.thinking_tail_feedback_freshness||{};
  const backlog=fresh.backlog_proof||{};
  const loop=data.thinking_tail_compaction_loop_status||{};
  const loopState=loop.current_rule_state||{};
  const canary=loop.canary||{};
  const savings=loop.savings||{};
  const quality=loop.quality||{};
  const outcome=loop.outcome_window||{};
  const loopFeedback=loop.managed_feedback||{};
  const blocker=data.top_blocker_reason||'none';
  const status=data.status||'unknown';
  text('managed-activation-status',status);
  text('managed-activation-detail',`${proof.status||'missing'} proof · ${num(feedback.queued)} queued · ${blocker}`);
  document.getElementById('managed-activation-status').className='value '+(status==='ready'?'money':(status==='no-proof'?'muted':'warn'));
  const privacy=data.privacy||{};
  const privacyBadge=(privacy.raw_request_bodies_included||privacy.raw_responses_included||privacy.payload_json_included)
    ?'<span class="badge err">raw included</span>'
    :'<span class="badge ok">metadata only</span>';
  const activationTbody=document.getElementById('managed-activation-tbody');
  if(activationTbody){
    activationTbody.innerHTML=`<tr>
      <td><span class="badge ${proof.status==='observed'?'ok':'miss'}">${esc(proof.status||'missing')}</span><div class="muted">${localTime(proof.latest_observed_at)}</div></td>
      <td>${esc(proof.decision_status||'-')}<div class="muted">${esc(proof.policy_id||proof.decision_id||'-')}</div></td>
      <td>${esc(proof.candidate_id||readiness.candidate_id||backlog.candidate_id||'-')}</td>
      <td><span class="badge ${readiness.ready?'ok':'miss'}">${esc(readiness.status||'missing')}</span><div class="muted">${esc((readiness.reason_codes||[]).join(', ')||'-')}</div></td>
      <td>${num(feedback.queued)} queued · ${num(feedback.sent)} sent<div class="muted">${num(feedback.retryable_error)} retryable · ${num(feedback.dropped_after_limit)} dropped</div></td>
      <td>${num(feedback.due||backlog.due)}</td>
      <td>${fmtSec(fresh.last_successful_drain_age_seconds)}</td>
      <td>${esc(blocker)}</td>
      <td>${privacyBadge}</td>
    </tr>`;
  }
  const loopPrivacy=loop.privacy||{};
  const loopPrivacyBadge=(loopPrivacy.raw_prompts_included||loopPrivacy.raw_messages_included||loopPrivacy.raw_request_bodies_included||loopPrivacy.raw_thinking_text_included||loopPrivacy.session_ids_included||loopPrivacy.file_paths_included)
    ?'<span class="badge err">raw included</span>'
    :'<span class="badge ok">metadata only</span>';
  const loopCls=loopState.state==='saving'||loopState.state==='ready-to-widen'?'ok':(loopState.state==='safety-stopped'||loopState.state==='feedback-blocked'?'err':'miss');
  const loopTbody=document.getElementById('thinking-tail-loop-tbody');
  if(loopTbody){
    loopTbody.innerHTML=`<tr>
      <td><span class="badge ${loopCls}">${esc(loopState.state||'unknown')}</span><div class="muted">${esc(loopState.top_blocker_reason||loopState.impact_status||'-')}</div></td>
      <td>${esc(loopState.policy_source||'-')}<div class="muted">target ${loopState.next_fraction_cap==null?'-':Math.round(Number(loopState.next_fraction_cap||0)*100)+'%'} · holdout ${loopState.holdout_fraction==null?'-':Math.round(Number(loopState.holdout_fraction||0)*100)+'%'}</div></td>
      <td>${num(canary.applied_count)} applied · ${num(canary.holdout_count)} holdout<div class="muted">${num(canary.safety_stop_count)} safety stopped · ${num(canary.skipped_count)} skipped</div></td>
      <td><span class="money">${money(savings.realized_savings_usd)}</span><div class="muted">projected ${money(savings.projected_savings_usd)} · ${num(savings.observed_saved_tokens)} tokens</div></td>
      <td>${Math.round(Number(quality.applied_minus_holdout_error_rate||0)*10000)/100}%</td>
      <td>${Math.round(Number(quality.applied_minus_holdout_retry_rate||0)*10000)/100}%</td>
      <td>${outcome.latest_outcome_at?localTime(outcome.latest_outcome_at):'-'}<div class="muted">last drain ${fmtSec(outcome.last_successful_drain_age_seconds)}</div></td>
      <td><span class="badge ${loopFeedback.status==='sent'?'ok':loopFeedback.status==='blocked'?'err':'miss'}">${esc(loopFeedback.status||'missing')}</span><div class="muted">${num(loopFeedback.pending)} pending · ${num(loopFeedback.sent)} sent · ${esc(loopFeedback.blocked_reason||'-')}</div></td>
      <td>${loopPrivacyBadge}</td>
    </tr>`;
  }
}
function renderWeekly(data){
  const totals=data.totals||{};
  text('week-units',num(totals.total_units));
  text('week-success',`${num(totals.successful_calls)} successful`);
  text('week-spend',money(totals.cost_est_usd));
  text('week-baseline',`${money(totals.cost_baseline_usd)} baseline`);
  text('week-savings',money(totals.savings_usd));
  text('week-tokens',num(totals.total_tokens));
  text('week-errors',`${num(totals.errors)} errors`);
  const managed=data.managed_feed||{};
  const managedState=managed.state||{};
  const managedWeek=managed.last_7_days||totals.managed_feed||{};
  text('week-managed-state',managedState.mode||'local_only');
  text('week-managed-detail',managedCompactText(managedWeek));
  document.getElementById('week-managed-state').className='value '+(managedState.server_calls_enabled?'money':'muted');
  document.getElementById('week-managed-tbody').innerHTML=managedSummaryRow('Last 7 days',managedState,managedWeek);
  renderRoutingConversion(managedWeek.routing_conversion||{});
  const rows=(data.days||[]).map(row=>`<tr>
    <td>${row.day}</td>
    <td>${num(row.total_units)}</td>
    <td>${num(row.provider_calls)}</td>
    <td>${num(row.codex_turns)}</td>
    <td>${num(row.successful_calls)}</td>
    <td>${num(row.errors)}</td>
    <td>${num(row.total_tokens)}</td>
    <td class="money">${money(row.cost_est_usd)}</td>
    <td>${money(row.cost_baseline_usd)}</td>
    <td class="money">${money(row.savings_usd)}</td>
  </tr>`).join('');
  document.getElementById('weekly-tbody').innerHTML=rows||'<tr><td colspan="10" class="muted">No weekly activity recorded yet.</td></tr>';
}
async function refresh(){
  try{
    const [today,weekly]=await Promise.all([
      getJson('/tokenclaw/stats'),
      getJson('/tokenclaw/stats/weekly'),
    ]);
    renderToday(today);
    renderWeekly(weekly);
    statusEl.textContent='updated '+new Date().toLocaleTimeString();
  }catch(error){
    statusEl.textContent='error loading dashboard';
    console.error(error);
  }
}
refresh();
setInterval(refresh,5000);
</script>
</body>
</html>"""
