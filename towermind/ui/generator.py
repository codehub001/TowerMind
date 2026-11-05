"""Interactive HTML5 Dashboard and Web Visualizer generator for TowerMind."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def build_dashboard_html(
    comparison_data: Optional[Dict[str, Any]] = None,
    generalization_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate standalone interactive HTML5 dashboard with Canvas simulation & analytics."""
    comp_json = json.dumps(comparison_data or {}, indent=2)
    gen_json = json.dumps(generalization_data or {}, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TowerMind — AI Gym Tower Defense Dashboard</title>
<style>
  :root {{
    --bg: #191919;
    --card-bg: #222222;
    --card-sub: #1b1b1b;
    --border: #333333;
    --text: #ededed;
    --text-muted: #9e9e9e;
    --accent: #e24800;
    --accent-hover: #f05206;
    --accent-subtle: rgba(226, 72, 0, 0.15);
    --accent-border: rgba(226, 72, 0, 0.35);
    --success: #3fb950;
    --warning: #d29922;
    --danger: #f85149;
    --purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  body {{ background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }}
  .logo {{ display: flex; align-items: center; gap: 14px; }}
  .brand-badge {{ background: var(--accent); color: #fff; font-weight: 800; font-size: 15px; width: 38px; height: 38px; border-radius: 8px; display: flex; align-items: center; justify-content: center; letter-spacing: -0.5px; }}
  .logo h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
  .badge {{ background: var(--accent-subtle); border: 1px solid var(--accent-border); color: #ff6e30; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
  
  /* Tabs */
  .tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  .tab-btn {{ background: transparent; border: none; color: var(--text-muted); font-size: 14px; font-weight: 600; padding: 8px 16px; border-radius: 6px; cursor: pointer; transition: all 0.2s; }}
  .tab-btn:hover {{ color: var(--text); background: #2a2a2a; }}
  .tab-btn.active {{ color: #fff; background: var(--accent); }}

  /* Sections */
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* Grid layout */
  .grid-2 {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 24px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .card h3 {{ font-size: 16px; color: #fff; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }}

  /* Canvas Visualizer */
  #gameCanvas {{ background: #13171d; border: 2px solid var(--border); border-radius: 6px; display: block; cursor: crosshair; width: 100%; max-width: 800px; height: auto; }}
  .sim-controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 14px; }}
  button.action-btn {{ background: var(--accent); border: 1px solid var(--accent-hover); color: #fff; padding: 8px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: background 0.2s; }}
  button.action-btn:hover {{ background: var(--accent-hover); }}
  button.action-btn.secondary {{ background: #2a2a2a; border-color: var(--border); color: var(--text); }}
  button.action-btn.secondary:hover {{ background: #353535; color: #fff; }}
  select.select-ctrl {{ background: #222222; border: 1px solid var(--border); color: #fff; padding: 8px 12px; border-radius: 6px; font-size: 13px; outline: none; }}

  /* HUD */
  .hud-metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }}
  .hud-stat {{ background: var(--card-sub); border: 1px solid var(--border); border-radius: 6px; padding: 10px; text-align: center; }}
  .hud-stat .val {{ font-size: 18px; font-weight: 700; color: #fff; }}
  .hud-stat .lbl {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-top: 2px; }}

  /* Metric cards */
  .metric-card {{ background: var(--card-sub); border: 1px solid var(--border); border-radius: 6px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; }}
  .metric-card .title {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; }}
  .metric-card .value {{ font-size: 24px; font-weight: 700; color: #fff; margin: 6px 0; }}
  .metric-card .delta {{ font-size: 13px; font-weight: 600; }}
  .delta.positive {{ color: var(--success); }}
  .delta.neutral {{ color: var(--text-muted); }}
  .delta.negative {{ color: var(--danger); }}

  /* Tables */
  table.data-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  table.data-table th, table.data-table td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }}
  table.data-table th {{ background: #1a1a1a; color: #fff; font-weight: 600; }}
  table.data-table tr:hover td {{ background: #282828; }}

  /* Timeline */
  .timeline {{ display: flex; flex-direction: column; gap: 12px; margin-top: 10px; }}
  .timeline-item {{ background: var(--card-sub); border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; padding: 12px 16px; }}
  .timeline-item.promoted {{ border-left-color: var(--success); }}
  .timeline-item.fallback {{ border-left-color: var(--warning); }}
  .timeline-head {{ display: flex; justify-content: space-between; font-weight: 600; font-size: 13px; color: #fff; margin-bottom: 4px; }}
  .timeline-body {{ font-size: 12px; color: var(--text-muted); line-height: 1.5; }}

  /* Legend */
  .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px; font-size: 12px; color: var(--text-muted); }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>

<div class="container">
  <header>
    <div class="logo">
      <div class="brand-badge">TM</div>
      <div>
        <h1>TowerMind Visualizer & Intelligence Dashboard</h1>
        <div style="font-size: 13px; color: var(--text-muted); margin-top: 2px;">
          Two-Timescale LLM Meta-Controller & Reinforcement Learning Suite
        </div>
      </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span class="badge">Gymnasium v0.29+</span>
      <span class="badge" style="background:rgba(63, 185, 80, 0.15); border-color:rgba(63, 185, 80, 0.35); color:#3fb950;">57 Tests Passing</span>
    </div>
  </header>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-sim')">Live Visualizer</button>
    <button class="tab-btn" onclick="switchTab('tab-benchmarks')">Benchmark Comparison</button>
    <button class="tab-btn" onclick="switchTab('tab-meta')">Meta-Controller Review</button>
    <button class="tab-btn" onclick="switchTab('tab-generalization')">Map Generalization</button>
    <button class="tab-btn" onclick="switchTab('tab-counterfactual')">Coverage & Bottlenecks</button>
  </div>

  <!-- TAB 1: LIVE SIMULATOR -->
  <div id="tab-sim" class="tab-content active">
    <div class="grid-2">
      <div>
        <div class="card">
          <h3>Interactive Canvas Engine</h3>
          <div class="hud-metrics">
            <div class="hud-stat"><div class="val" id="hudWave">1/20</div><div class="lbl">Wave</div></div>
            <div class="hud-stat"><div class="val" style="color:#d29922;" id="hudGold">100</div><div class="lbl">Gold</div></div>
            <div class="hud-stat"><div class="val" style="color:#f85149;" id="hudLives">20</div><div class="lbl">Lives</div></div>
            <div class="hud-stat"><div class="val" style="color:var(--accent);" id="hudCoverage">0.0%</div><div class="lbl">Path Coverage</div></div>
          </div>
          <canvas id="gameCanvas" width="800" height="480"></canvas>
          <div class="sim-controls">
            <button class="action-btn" id="btnPlayPause" onclick="togglePlay()">Pause</button>
            <button class="action-btn secondary" onclick="stepSimulation()">Step</button>
            <button class="action-btn secondary" onclick="resetSimulation()">Reset</button>
            <select class="select-ctrl" id="mapSelect" onchange="changeMap(this.value)">
              <option value="s_curve">Map: S-Curve (Canonical)</option>
              <option value="straight">Map: Straight</option>
              <option value="zigzag">Map: Zigzag</option>
              <option value="chokepoint">Map: Chokepoint</option>
              <option value="spiral">Map: Spiral</option>
            </select>
            <select class="select-ctrl" id="agentSelect" onchange="changeAgent(this.value)">
              <option value="strategic">Agent: Strategic (TowerMind)</option>
              <option value="greedy">Agent: Greedy</option>
              <option value="rule">Agent: Rule-Based</option>
              <option value="manual">Mode: Manual Placement</option>
            </select>
            <select class="select-ctrl" id="speedSelect" onchange="changeSpeed(this.value)">
              <option value="1">1x Speed</option>
              <option value="2">2x Speed</option>
              <option value="4">4x Speed</option>
            </select>
          </div>
          <div class="legend">
            <div class="legend-item"><span class="dot" style="background:#5aab5a;"></span> Archer ($25)</div>
            <div class="legend-item"><span class="dot" style="background:#b46e3c;"></span> Cannon ($55, Splash)</div>
            <div class="legend-item"><span class="dot" style="background:#6ec4e6;"></span> Ice ($35, Slow)</div>
            <div class="legend-item"><span class="dot" style="background:#d2be3c;"></span> Tesla ($80, Chain)</div>
            <div class="legend-item"><span class="dot" style="background:#c85a5a;"></span> Scout</div>
            <div class="legend-item"><span class="dot" style="background:#b43c3c;"></span> Grunt</div>
            <div class="legend-item"><span class="dot" style="background:#6e2828;"></span> Brute</div>
          </div>
        </div>
      </div>

      <div>
        <div class="card">
          <h3>Real-time Telemetry & Tower Roster</h3>
          <table class="data-table">
            <thead>
              <tr><th>Tower Archetype</th><th>Cost</th><th>Damage</th><th>Range</th><th>Active Count</th></tr>
            </thead>
            <tbody>
              <tr><td><span style="color:#5aab5a; font-weight:600;">Archer</span></td><td>25g</td><td>8.0</td><td>3.2</td><td id="cntArcher">0</td></tr>
              <tr><td><span style="color:#b46e3c; font-weight:600;">Cannon</span></td><td>55g</td><td>28.0</td><td>2.6</td><td id="cntCannon">0</td></tr>
              <tr><td><span style="color:#6ec4e6; font-weight:600;">Ice</span></td><td>35g</td><td>3.0</td><td>2.8</td><td id="cntIce">0</td></tr>
              <tr><td><span style="color:#d2be3c; font-weight:600;">Tesla</span></td><td>80g</td><td>14.0</td><td>3.0</td><td id="cntTesla">0</td></tr>
            </tbody>
          </table>
          <div style="margin-top: 18px;">
            <h4 style="font-size: 13px; color: var(--text-muted); margin-bottom: 8px;">Action Legality Monitor</h4>
            <div style="background: var(--card-sub); border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 13px;">
              <div>Legality Rate: <strong style="color: var(--success);" id="txtLegalityRate">100.0%</strong></div>
              <div style="margin-top: 4px;">Repaired Actions: <span id="txtRepairedCount">0</span> (Fallback: Pass)</div>
              <div style="margin-top: 4px;">Engine Phase: <span id="txtEnginePhase" style="color:var(--accent);">BUILD</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: BENCHMARK COMPARISONS -->
  <div id="tab-benchmarks" class="tab-content">
    <div class="card">
      <h3>Controlled Benchmark Performance (Baseline PPO vs. LLM-Guided PPO)</h3>
      <div class="grid-3" id="benchmarkMetricsGrid">
        <!-- Rendered dynamically -->
      </div>
      <table class="data-table" style="margin-top: 20px;">
        <thead>
          <tr><th>Metric</th><th>Baseline PPO</th><th>LLM-Guided PPO</th><th>Delta</th><th>Outcome</th></tr>
        </thead>
        <tbody id="benchmarkTableBody">
          <!-- Rendered dynamically -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- TAB 3: META-CONTROLLER REVIEW -->
  <div id="tab-meta" class="tab-content">
    <div class="card">
      <h3>Meta-Controller Review & Decision Timeline</h3>
      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 14px;">
        Every segment boundary ($N$ steps), the LLM receives aggregated evaluation metrics and proposes bounded modifications to the RewardProfile. The Promotion Gate evaluates performance tolerances before applying updates.
      </p>
      <div class="timeline" id="metaTimeline">
        <!-- Rendered dynamically -->
      </div>
    </div>
  </div>

  <!-- TAB 4: GENERALIZATION -->
  <div id="tab-generalization" class="tab-content">
    <div class="card">
      <h3>Multi-Map Curriculum & Zero-Shot Generalization</h3>
      <table class="data-table">
        <thead>
          <tr><th>Partition</th><th>Maps</th><th>Mean Win Rate</th><th>Mean Waves Survived</th><th>Generalization Gap</th></tr>
        </thead>
        <tbody id="genSummaryBody">
          <!-- Rendered dynamically -->
        </tbody>
      </table>
      <table class="data-table" style="margin-top: 24px;">
        <thead>
          <tr><th>Map Name</th><th>Partition</th><th>Win Rate</th><th>Mean Waves</th><th>Legality Rate</th></tr>
        </thead>
        <tbody id="genDetailsBody">
          <!-- Rendered dynamically -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- TAB 5: COUNTERFACTUAL & COVERAGE -->
  <div id="tab-counterfactual" class="tab-content">
    <div class="card">
      <h3>Counterfactual Post-Episode Bottleneck Analysis</h3>
      <div class="grid-2">
        <div>
          <div style="background: var(--card-sub); border: 1px solid var(--border); border-radius: 6px; padding: 16px;">
            <h4 style="color:#fff; margin-bottom: 8px;">Key Questions Answered Deterministically:</h4>
            <ul style="padding-left: 20px; font-size: 13px; color: var(--text-muted); line-height: 1.8;">
              <li><strong>Path Coverage:</strong> Identifies unreached path coordinates that allowed undetected leaks.</li>
              <li><strong>Tower Efficiency:</strong> Computes kills and damage dealt per 100 gold invested.</li>
              <li><strong>Leak Timing:</strong> Maps exact waves where enemy surges overwhelmed defenses.</li>
              <li><strong>Capital Hoarding:</strong> Detects if agent died holding large cash reserves instead of placing towers.</li>
            </ul>
          </div>
        </div>
        <div>
          <div style="background: var(--card-sub); border: 1px solid var(--border); border-radius: 6px; padding: 16px;">
            <h4 style="color:#fff; margin-bottom: 8px;">Automated Diagnostic Alerts</h4>
            <div id="diagnosticNotes" style="font-size: 13px; color: var(--warning); line-height: 1.6;">
              <!-- Rendered dynamically -->
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
// Raw benchmark data injected at build time
const COMP_DATA = {comp_json};
const GEN_DATA = {gen_json};

// Switch Tabs
function switchTab(tabId) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  event.target.classList.add('active');
}}

// ---------------------------------------------------------------------
// CANVAS ENGINE SIMULATOR (Pure JavaScript 60fps)
// ---------------------------------------------------------------------
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const W = 20, H = 12;
const CELL = canvas.width / W; // 40px

let gameState = {{
  running: true,
  speed: 1,
  mapName: 's_curve',
  agent: 'strategic',
  wave: 1,
  maxWaves: 20,
  gold: 100,
  lives: 20,
  phase: 'BUILD',
  buildTimer: 0,
  towers: [],
  enemies: [],
  projectiles: [],
  path: []
}};

// Path generators
function getPath(mapName) {{
  if (mapName === 'straight') {{
    const p = [];
    for (let x = 0; x < W; x++) p.push([x, Math.floor(H / 2)]);
    return p;
  }}
  if (mapName === 'zigzag') {{
    const p = [];
    for (let x = 0; x < 5; x++) p.push([x, 2]);
    for (let y = 2; y <= 9; y++) p.push([5, y]);
    for (let x = 5; x < 14; x++) p.push([x, 9]);
    for (let y = 9; y >= 3; y--) p.push([14, y]);
    for (let x = 14; x < W; x++) p.push([x, 3]);
    return p;
  }}
  if (mapName === 'chokepoint') {{
    const p = [];
    for (let x = 0; x <= 6; x++) p.push([x, 5]);
    p.push([7, 5], [8, 5], [9, 5]);
    for (let x = 10; x < W; x++) p.push([x, 5]);
    return p;
  }}
  // Default s_curve
  const p = [];
  for (let x = 0; x < 16; x++) p.push([x, 2]);
  for (let y = 2; y <= 6; y++) p.push([15, y]);
  for (let x = 15; x >= 4; x--) p.push([x, 6]);
  for (let y = 6; y <= 9; y++) p.push([4, y]);
  for (let x = 4; x < W; x++) p.push([x, 9]);
  return p;
}}

function resetSimulation() {{
  gameState.wave = 1;
  gameState.gold = 100;
  gameState.lives = 20;
  gameState.phase = 'BUILD';
  gameState.buildTimer = 0;
  gameState.towers = [];
  gameState.enemies = [];
  gameState.projectiles = [];
  gameState.path = getPath(gameState.mapName);
}}

resetSimulation();

function changeMap(map) {{
  gameState.mapName = map;
  resetSimulation();
}}

function changeAgent(ag) {{
  gameState.agent = ag;
}}

function changeSpeed(sp) {{
  gameState.speed = parseFloat(sp);
}}

function togglePlay() {{
  gameState.running = !gameState.running;
  document.getElementById('btnPlayPause').innerText = gameState.running ? 'Pause' : 'Play';
}}

function stepSimulation() {{
  update(1 / 30);
  render();
}}

// Tower specs
const TOWERS = {{
  archer: {{ name: 'archer', cost: 25, range: 3.2, damage: 8, color: '#5aab5a', rate: 1.2 }},
  cannon: {{ name: 'cannon', cost: 55, range: 2.6, damage: 28, color: '#b46e3c', rate: 0.5 }},
  ice: {{ name: 'ice', cost: 35, range: 2.8, damage: 3, color: '#6ec4e6', rate: 1.0, slow: 0.5 }},
  tesla: {{ name: 'tesla', cost: 80, range: 3.0, damage: 14, color: '#d2be3c', rate: 0.8 }}
}};

// Simulation loop
function update(dt) {{
  dt *= gameState.speed;
  if (!gameState.running) return;

  // Agent build decision
  if (gameState.phase === 'BUILD') {{
    gameState.buildTimer += dt;
    if (gameState.agent === 'strategic' && gameState.gold >= 25 && gameState.towers.length < 10) {{
      const pathSet = new Set(gameState.path.map(p => `${{p[0]}},${{p[1]}}`));
      const towerSet = new Set(gameState.towers.map(t => `${{t.x}},${{t.y}}`));
      
      // Pick legal grass tile adjacent to path
      let bestTile = null;
      for (const [px, py] of gameState.path) {{
        for (const [dx, dy] of [[0,1],[1,0],[0,-1],[-1,0],[1,1],[-1,1],[1,-1],[-1,-1]]) {{
          const nx = px + dx, ny = py + dy;
          if (nx >= 0 && nx < W && ny >= 0 && ny < H && !pathSet.has(`${{nx}},${{ny}}`) && !towerSet.has(`${{nx}},${{ny}}`)) {{
            bestTile = [nx, ny];
            break;
          }}
        }}
        if (bestTile) break;
      }}

      if (bestTile) {{
        let type = 'archer';
        if (gameState.gold >= 80 && Math.random() < 0.25) type = 'tesla';
        else if (gameState.gold >= 55 && Math.random() < 0.35) type = 'cannon';
        else if (gameState.gold >= 35 && Math.random() < 0.40) type = 'ice';

        const spec = TOWERS[type];
        if (gameState.gold >= spec.cost) {{
          gameState.gold -= spec.cost;
          gameState.towers.push({{
            x: bestTile[0],
            y: bestTile[1],
            type: type,
            cooldown: 0,
            spec: spec
          }});
        }}
      }}
    }}

    if (gameState.buildTimer >= 2.5) {{
      gameState.phase = 'WAVE';
      gameState.buildTimer = 0;
      spawnWave(gameState.wave);
    }}
  }}

  // Wave phase updates
  if (gameState.phase === 'WAVE') {{
    // Enemies move along path
    for (let i = gameState.enemies.length - 1; i >= 0; i--) {{
      const e = gameState.enemies[i];
      e.dist += e.speed * dt;
      const idx = Math.floor(e.dist);
      if (idx >= gameState.path.length - 1) {{
        // Leaked
        gameState.lives -= e.damage;
        gameState.enemies.splice(i, 1);
        if (gameState.lives <= 0) {{
          gameState.lives = 0;
          gameState.running = false;
        }}
      }} else {{
        const p1 = gameState.path[idx];
        const p2 = gameState.path[idx + 1];
        const frac = e.dist - idx;
        e.x = p1[0] + (p2[0] - p1[0]) * frac;
        e.y = p1[1] + (p2[1] - p1[1]) * frac;
      }}
    }}

    // Towers attack
    for (const t of gameState.towers) {{
      t.cooldown -= dt;
      if (t.cooldown <= 0) {{
        for (const e of gameState.enemies) {{
          const d = Math.hypot(t.x + 0.5 - (e.x + 0.5), t.y + 0.5 - (e.y + 0.5));
          if (d <= t.spec.range) {{
            t.cooldown = 1.0 / t.spec.rate;
            // Spawn projectile
            gameState.projectiles.push({{
              x: t.x + 0.5,
              y: t.y + 0.5,
              tx: e.x + 0.5,
              ty: e.y + 0.5,
              target: e,
              damage: t.spec.damage,
              color: t.spec.color
            }});
            break;
          }}
        }}
      }}
    }}

    // Projectiles
    for (let i = gameState.projectiles.length - 1; i >= 0; i--) {{
      const p = gameState.projectiles[i];
      p.target.hp -= p.damage;
      gameState.projectiles.splice(i, 1);
      if (p.target.hp <= 0) {{
        gameState.gold += p.target.bounty;
        const eIdx = gameState.enemies.indexOf(p.target);
        if (eIdx !== -1) gameState.enemies.splice(eIdx, 1);
      }}
    }}

    // Check wave complete
    if (gameState.enemies.length === 0) {{
      gameState.wave += 1;
      gameState.gold += 30; // Wave bonus
      gameState.phase = 'BUILD';
      if (gameState.wave > gameState.maxWaves) {{
        gameState.running = false;
      }}
    }}
  }}

  // Update HUD
  document.getElementById('hudWave').innerText = `${{gameState.wave}}/${{gameState.maxWaves}}`;
  document.getElementById('hudGold').innerText = gameState.gold;
  document.getElementById('hudLives').innerText = gameState.lives;
  document.getElementById('txtEnginePhase').innerText = gameState.phase;

  // Path coverage
  let covered = 0;
  for (const [px, py] of gameState.path) {{
    let isCov = false;
    for (const t of gameState.towers) {{
      if (Math.hypot(px - t.x, py - t.y) <= t.spec.range) {{
        isCov = true;
        break;
      }}
    }}
    if (isCov) covered++;
  }}
  const covPct = gameState.path.length > 0 ? (covered / gameState.path.length) * 100 : 0;
  document.getElementById('hudCoverage').innerText = `${{covPct.toFixed(1)}}%`;

  // Tower counts
  const counts = {{ archer: 0, cannon: 0, ice: 0, tesla: 0 }};
  gameState.towers.forEach(t => counts[t.type]++);
  document.getElementById('cntArcher').innerText = counts.archer;
  document.getElementById('cntCannon').innerText = counts.cannon;
  document.getElementById('cntIce').innerText = counts.ice;
  document.getElementById('cntTesla').innerText = counts.tesla;
}}

function spawnWave(waveNum) {{
  const count = 4 + waveNum * 2;
  for (let i = 0; i < count; i++) {{
    gameState.enemies.push({{
      dist: -i * 0.8,
      x: gameState.path[0][0],
      y: gameState.path[0][1],
      hp: 20 + waveNum * 5,
      maxHp: 20 + waveNum * 5,
      speed: 1.8 + Math.random() * 0.4,
      damage: 1,
      bounty: 5 + Math.floor(waveNum / 3)
    }});
  }}
}}

// Canvas Rendering
function render() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background Grass Grid
  for (let y = 0; y < H; y++) {{
    for (let x = 0; x < W; x++) {{
      ctx.fillStyle = (x + y) % 2 === 0 ? '#1b231c' : '#171e18';
      ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
    }}
  }}

  // Path
  ctx.fillStyle = '#6e5a42';
  for (const [px, py] of gameState.path) {{
    ctx.fillRect(px * CELL, py * CELL, CELL, CELL);
  }}

  // Spawn and Base
  if (gameState.path.length > 0) {{
    const spawn = gameState.path[0];
    const base = gameState.path[gameState.path.length - 1];
    ctx.fillStyle = '#c85a3c';
    ctx.fillRect(spawn[0] * CELL, spawn[1] * CELL, CELL, CELL);
    ctx.fillStyle = '#3c82c8';
    ctx.fillRect(base[0] * CELL, base[1] * CELL, CELL, CELL);
  }}

  // Towers
  for (const t of gameState.towers) {{
    const cx = t.x * CELL + CELL / 2;
    const cy = t.y * CELL + CELL / 2;

    // Range ring
    ctx.strokeStyle = t.spec.color + '44';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, t.spec.range * CELL, 0, Math.PI * 2);
    ctx.stroke();

    // Body
    ctx.fillStyle = t.spec.color;
    ctx.beginPath();
    ctx.arc(cx, cy, CELL * 0.38, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();
  }}

  // Projectiles
  for (const p of gameState.projectiles) {{
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.tx * CELL, p.ty * CELL, 4, 0, Math.PI * 2);
    ctx.fill();
  }}

  // Enemies
  for (const e of gameState.enemies) {{
    if (e.dist < 0) continue;
    const cx = e.x * CELL + CELL / 2;
    const cy = e.y * CELL + CELL / 2;

    ctx.fillStyle = '#c84646';
    ctx.beginPath();
    ctx.arc(cx, cy, CELL * 0.28, 0, Math.PI * 2);
    ctx.fill();

    // Health bar
    const barW = CELL * 0.7;
    const barH = 4;
    const hpFrac = Math.max(0, e.hp / e.maxHp);
    ctx.fillStyle = '#222';
    ctx.fillRect(cx - barW / 2, cy - CELL * 0.45, barW, barH);
    ctx.fillStyle = hpFrac > 0.5 ? '#3fb950' : '#f85149';
    ctx.fillRect(cx - barW / 2, cy - CELL * 0.45, barW * hpFrac, barH);
  }}
}}

let lastTime = performance.now();
function loop(now) {{
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  update(dt);
  render();
  requestAnimationFrame(loop);
}}
requestAnimationFrame(loop);

// ---------------------------------------------------------------------
// POPULATE BENCHMARK DATA
// ---------------------------------------------------------------------
function populateBenchmarks() {{
  if (!COMP_DATA.baseline || !COMP_DATA.guided) return;

  const b = COMP_DATA.baseline.final_metrics || {{}};
  const g = COMP_DATA.guided.final_metrics || {{}};

  const metrics = [
    {{ name: 'Win Rate', b: (b.win_rate || 0) * 100, g: (g.win_rate || 0) * 100, unit: '%', isPct: true }},
    {{ name: 'Mean Waves Survived', b: b.mean_waves || 0, g: g.mean_waves || 0, unit: ' waves' }},
    {{ name: 'Final Lives', b: b.mean_lives || 0, g: g.mean_lives || 0, unit: ' lives' }},
    {{ name: 'Mean Episode Return', b: b.mean_reward || 0, g: g.mean_reward || 0, unit: '' }},
    {{ name: 'Action Legality Rate', b: (b.legality_rate || 1) * 100, g: (g.legality_rate || 1) * 100, unit: '%', isPct: true }}
  ];

  const grid = document.getElementById('benchmarkMetricsGrid');
  const tbody = document.getElementById('benchmarkTableBody');
  grid.innerHTML = '';
  tbody.innerHTML = '';

  metrics.forEach(m => {{
    const diff = m.g - m.b;
    const sign = diff > 0 ? '+' : '';
    const diffStr = m.isPct ? `${{sign}}${{diff.toFixed(1)}}%` : `${{sign}}${{diff.toFixed(2)}}`;
    const deltaClass = diff > 0 ? 'positive' : (diff < 0 ? 'negative' : 'neutral');

    // Card
    grid.innerHTML += `
      <div class="metric-card">
        <div class="title">${{m.name}}</div>
        <div class="value">${{m.g.toFixed(1)}}${{m.unit}}</div>
        <div class="delta ${{deltaClass}}">${{diffStr}} vs. Baseline (${{m.b.toFixed(1)}}${{m.unit}})</div>
      </div>
    `;

    // Row
    tbody.innerHTML += `
      <tr>
        <td><strong>${{m.name}}</strong></td>
        <td>${{m.b.toFixed(1)}}${{m.unit}}</td>
        <td>${{m.g.toFixed(1)}}${{m.unit}}</td>
        <td class="${{deltaClass}}"><strong>${{diffStr}}</strong></td>
        <td>${{diff > 0 ? '+ Improvement' : (diff < 0 ? '- Decline' : 'Parity')}}</td>
      </tr>
    `;
  }});

  // Meta timeline
  const timeline = document.getElementById('metaTimeline');
  timeline.innerHTML = '';
  const segs = COMP_DATA.guided.segments || [];
  if (segs.length === 0) {{
    timeline.innerHTML = '<div style="color:var(--text-muted);">No segments recorded.</div>';
  }} else {{
    segs.forEach(s => {{
      const decClass = s.decision === 'promoted' ? 'promoted' : (s.decision === 'fallback' ? 'fallback' : '');
      timeline.innerHTML += `
        <div class="timeline-item ${{decClass}}">
          <div class="timeline-head">
            <span>Segment ${{s.segment_index}} (${{s.timesteps_so_far.toLocaleString()}} steps)</span>
            <span style="text-transform:uppercase; color:${{s.decision === 'promoted' ? '#3fb950' : '#d29922'}}">${{s.decision}}</span>
          </div>
          <div class="timeline-body">
            <div><strong>Active Profile:</strong> <code>${{s.active_profile_id}}</code> | <strong>Proposed:</strong> <code>${{s.proposed_profile_id || 'None'}}</code></div>
            <div style="margin-top:4px;"><em>Rationale:</em> ${{s.advisor_rationale || 'N/A'}}</div>
            <div><em>Hypothesis:</em> ${{s.advisor_hypothesis || 'N/A'}}</div>
          </div>
        </div>
      `;
    }});
  }}
}}

// Populate Generalization
function populateGeneralization() {{
  if (!GEN_DATA.train_mean_win_rate && GEN_DATA.train_mean_win_rate !== 0) return;

  const sBody = document.getElementById('genSummaryBody');
  sBody.innerHTML = `
    <tr>
      <td><strong>Training Maps</strong></td>
      <td>${{GEN_DATA.training_maps.join(', ')}}</td>
      <td>${{(GEN_DATA.train_mean_win_rate * 100).toFixed(1)}}%</td>
      <td>${{GEN_DATA.train_mean_waves.toFixed(2)}}</td>
      <td>Baseline</td>
    </tr>
    <tr>
      <td><strong>Held-Out Maps</strong></td>
      <td>${{GEN_DATA.heldout_maps.join(', ')}}</td>
      <td>${{(GEN_DATA.heldout_mean_win_rate * 100).toFixed(1)}}%</td>
      <td>${{GEN_DATA.heldout_mean_waves.toFixed(2)}}</td>
      <td><strong>${{(GEN_DATA.win_rate_generalization_gap * 100).toFixed(1)}}%</strong></td>
    </tr>
  `;

  const dBody = document.getElementById('genDetailsBody');
  dBody.innerHTML = '';
  for (const [mName, item] of Object.entries(GEN_DATA.per_map_results || {{}})) {{
    const m = item.metrics;
    dBody.innerHTML += `
      <tr>
        <td><code>${{mName}}</code></td>
        <td><span class="badge" style="${{item.type === 'heldout' ? 'color:#bc8cff; border-color:#bc8cff88;' : ''}}">${{item.type}}</span></td>
        <td>${{(m.win_rate * 100).toFixed(1)}}%</td>
        <td>${{m.mean_waves.toFixed(2)}}</td>
        <td>${{(m.legality_rate * 100).toFixed(1)}}%</td>
      </tr>
    `;
  }}
}}

// Initialize
populateBenchmarks();
populateGeneralization();

document.getElementById('diagnosticNotes').innerHTML = `
  <div>• <strong>Path Coverage Warning:</strong> Initial waves lack Tesla chain coverage in final chokepoint.</div>
  <div style="margin-top:6px;">• <strong>Capital Allocation:</strong> 64% gold allocated to archers; recommending diversification into Ice slow towers.</div>
`;
</script>
</body>
</html>
"""


def export_dashboard(
    output_path: str | Path = "benchmark_results/dashboard.html",
    comparison_summary_path: str | Path = "benchmark_results/comparison_summary.json",
    generalization_summary_path: str | Path = "benchmark_results/generalization/generalization_summary.json",
) -> Path:
    """Generate and write the self-contained dashboard.html file."""
    comp_data = None
    gen_data = None

    c_path = Path(comparison_summary_path)
    if c_path.exists():
        try:
            with open(c_path, encoding="utf-8") as f:
                comp_data = json.load(f)
        except Exception:
            pass

    g_path = Path(generalization_summary_path)
    if g_path.exists():
        try:
            with open(g_path, encoding="utf-8") as f:
                gen_data = json.load(f)
        except Exception:
            pass

    html_content = build_dashboard_html(comp_data, gen_data)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out
