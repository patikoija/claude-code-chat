const express = require('express');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const os = require('os');

const OLLAMA_BASE = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const OLLAMA_HOST_SSH = process.env.OLLAMA_HOST_SSH || 'localhost';
const OLLAMA_SESSIONS_DIR = process.env.OLLAMA_SESSIONS_DIR || path.join(__dirname, 'ollama-sessions');

// ============================================================
// VRAM guard — prevent gibberish from GPU memory pressure
// ============================================================

// Known VRAM requirements per model (MB), measured on RDNA 4 9060 XT
const MODEL_VRAM_MB = {
  'phi4-mini:3.8b': 8300,
  'llama3.1:8b': 10300,
  'deepseek-coder-v2:16b': 18700,
};

// Models ordered smallest to largest for auto-downgrade
const MODELS_BY_SIZE = [
  'phi4-mini:3.8b',
  'llama3.1:8b',
  'deepseek-coder-v2:16b',
];

// Check which models are already loaded in Ollama (they don't need extra VRAM)
async function getLoadedModels() {
  try {
    const data = await new Promise((resolve, reject) => {
      http.get(`${OLLAMA_BASE}/api/ps`, (r) => {
        let body = '';
        r.on('data', chunk => body += chunk);
        r.on('end', () => resolve(body));
        r.on('error', reject);
      }).on('error', reject);
    });
    const parsed = JSON.parse(data);
    return (parsed.models || []).map(m => m.name);
  } catch (e) {
    return [];
  }
}

// Read free VRAM from the GPU host via sysfs over SSH
function getFreeVramMB() {
  return new Promise((resolve) => {
    const proc = spawn('ssh', [OLLAMA_HOST_SSH,
      'echo $(( ($(cat /sys/class/drm/card*/device/mem_info_vram_total) - $(cat /sys/class/drm/card*/device/mem_info_vram_used)) / 1048576 ))'],
      { timeout: 5000 });
    let out = '';
    proc.stdout.on('data', d => out += d.toString());
    proc.on('close', (code) => {
      const mb = parseInt(out.trim(), 10);
      resolve(code === 0 && !isNaN(mb) ? mb : null);
    });
    proc.on('error', () => resolve(null));
  });
}

// Returns { ok, model, warning } — may downgrade model or reject
async function vramCheck(requestedModel) {
  const needed = MODEL_VRAM_MB[requestedModel];
  if (!needed) {
    // Unknown model — let it through, Ollama will handle errors
    return { ok: true, model: requestedModel };
  }

  // If the model is already loaded, no extra VRAM needed
  const loaded = await getLoadedModels();
  if (loaded.includes(requestedModel)) {
    return { ok: true, model: requestedModel };
  }

  const freeMB = await getFreeVramMB();
  if (freeMB === null) {
    // Can't read VRAM — let it through rather than block
    console.log('[vram-guard] Could not read VRAM, allowing request');
    return { ok: true, model: requestedModel };
  }

  console.log(`[vram-guard] Free VRAM: ${freeMB} MB, requested model ${requestedModel} needs ${needed} MB`);

  if (freeMB >= needed) {
    return { ok: true, model: requestedModel };
  }

  // Not enough VRAM — find the largest model that fits
  for (let i = MODELS_BY_SIZE.length - 1; i >= 0; i--) {
    const alt = MODELS_BY_SIZE[i];
    const altNeeded = MODEL_VRAM_MB[alt];
    if (freeMB >= altNeeded) {
      console.log(`[vram-guard] Downgrading ${requestedModel} -> ${alt} (${freeMB} MB free, need ${needed} MB)`);
      return {
        ok: true,
        model: alt,
        warning: `Not enough VRAM for ${requestedModel} (${freeMB} MB free, need ${needed} MB). Using ${alt} instead.`,
      };
    }
  }

  // Nothing fits
  console.log(`[vram-guard] No model fits in ${freeMB} MB free VRAM`);
  return {
    ok: false,
    model: requestedModel,
    warning: `Not enough GPU memory for any model (${freeMB} MB free). The GPU may be busy with another application. Try again later.`,
  };
}

const app = express();
const PORT = process.env.PORT || 3200;
const HOME_DIR = process.env.CLAUDE_CHAT_HOME || os.homedir();

// Ensure Ollama sessions directory exists
if (!fs.existsSync(OLLAMA_SESSIONS_DIR)) {
  fs.mkdirSync(OLLAMA_SESSIONS_DIR, { recursive: true });
}

// ============================================================
// Process management — processes persist across client disconnects
// ============================================================

// processId → { claude, buffer, finished, sessionId, listeners, startTime, exitCode }
const activeProcesses = new Map();

// sessionId → processId (lookup for reconnection)
const sessionToProcess = new Map();

let nextProcessId = 1;

// ============================================================
// Message queue — limits concurrent Claude processes to avoid rate limits
// ============================================================

const QUEUE_CONCURRENCY = parseInt(process.env.QUEUE_CONCURRENCY || '3', 10);
let activeCount = 0;
const messageQueue = []; // [{message, sessionId, res}]

function onProcessComplete() {
  if (messageQueue.length === 0) {
    activeCount = Math.max(0, activeCount - 1);
    return;
  }

  // Reuse the freed slot immediately for the next queued job
  const job = messageQueue.shift();

  // Update positions for remaining queued jobs
  messageQueue.forEach((j, i) => {
    try {
      j.res.write(`data: ${JSON.stringify({ type: 'queued', position: i + 1 })}\n\n`);
    } catch (e) { /* client disconnected */ }
  });

  // Tell this job it's now processing
  let skipJob = false;
  try {
    job.res.write(`data: ${JSON.stringify({ type: 'queue_start' })}\n\n`);
  } catch (e) {
    // Client disconnected before their turn — skip to next
    skipJob = true;
  }

  if (skipJob) {
    onProcessComplete(); // Recurse to get next valid job
    return;
  }

  const proc = createProcess(job.message, job.sessionId, onProcessComplete);
  addListener(proc, job.res);
  console.log(`Dequeued message for session ${job.sessionId || '(new)'}, ${messageQueue.length} remaining in queue`);
}

function enqueueOrProcess(message, sessionId, res) {
  if (activeCount < QUEUE_CONCURRENCY) {
    activeCount++;
    const proc = createProcess(message, sessionId, onProcessComplete);
    addListener(proc, res);
    console.log(`Started process ${proc.processId} for session ${sessionId || '(new)'}`);
  } else {
    const position = messageQueue.length + 1;
    messageQueue.push({ message, sessionId, res });
    console.log(`Message queued at position ${position} (session ${sessionId || 'new'})`);
    try {
      res.write(`data: ${JSON.stringify({ type: 'queued', position })}\n\n`);
    } catch (e) { /* ignore */ }

    // Remove from queue if client disconnects while waiting
    res.on('close', () => {
      const idx = messageQueue.findIndex(j => j.res === res);
      if (idx !== -1) {
        messageQueue.splice(idx, 1);
        console.log(`Queued message at position ${idx + 1} removed (client disconnected)`);
        // Rebroadcast updated positions to remaining jobs
        messageQueue.forEach((j, i) => {
          try {
            j.res.write(`data: ${JSON.stringify({ type: 'queued', position: i + 1 })}\n\n`);
          } catch (e) { /* ignore */ }
        });
      }
    });
  }
}

// Buffer limits to prevent memory exhaustion
const MAX_BUFFER_EVENTS = 10000;  // Max events to keep in memory
const BUFFER_DISK_PATH = '/tmp/claude-chat-buffers';

// Ensure buffer directory exists
if (!fs.existsSync(BUFFER_DISK_PATH)) {
  fs.mkdirSync(BUFFER_DISK_PATH, { recursive: true });
}

function createProcess(message, sessionId, onFinish) {
  const processId = `proc-${nextProcessId++}`;

  const args = [
    '-p', message,
    '--output-format', 'stream-json',
    '--verbose',
    '--permission-mode', 'bypassPermissions',
    '--model', 'sonnet',
  ];

  if (sessionId) {
    args.push('--resume', sessionId);
  }

  const claude = spawn('claude', args, {
    cwd: HOME_DIR,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: {
      ...process.env,
      HOME: HOME_DIR,
      PATH: process.env.PATH || '/usr/local/bin:/usr/bin:/bin',
    },
  });

  const proc = {
    processId,
    claude,
    buffer: [],
    finished: false,
    sessionId: sessionId || null,
    listeners: new Set(),
    startTime: Date.now(),
    exitCode: null,
  };

  activeProcesses.set(processId, proc);
  if (sessionId) {
    sessionToProcess.set(sessionId, processId);
  }

  // Emit initial event
  const startEvent = { type: 'process_start', processId };
  broadcastEvent(proc, startEvent);

  let stdoutBuffer = '';

  claude.stdout.on('data', (data) => {
    stdoutBuffer += data.toString();
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop();

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line);

        // Capture session ID from result events
        if (event.type === 'result' && event.session_id && !proc.sessionId) {
          proc.sessionId = event.session_id;
          sessionToProcess.set(event.session_id, processId);
        }

        broadcastEvent(proc, event);
      } catch (e) {
        // skip invalid JSON
      }
    }
  });

  claude.stderr.on('data', (data) => {
    const msg = data.toString();
    broadcastEvent(proc, { type: 'system', message: msg });
  });

  claude.on('close', (code) => {
    // Process remaining stdout buffer
    if (stdoutBuffer.trim()) {
      try {
        const event = JSON.parse(stdoutBuffer);
        if (event.type === 'result' && event.session_id && !proc.sessionId) {
          proc.sessionId = event.session_id;
          sessionToProcess.set(event.session_id, processId);
        }
        broadcastEvent(proc, event);
      } catch (e) {
        // ignore
      }
    }

    proc.finished = true;
    proc.exitCode = code;
    broadcastEvent(proc, { type: 'done', exitCode: code });

    // Close all listener connections
    for (const listener of proc.listeners) {
      listener.end();
    }
    proc.listeners.clear();

    // Schedule cleanup after 5 minutes
    setTimeout(() => cleanupProcess(processId), 5 * 60 * 1000);

    // Release the queue slot
    if (onFinish) onFinish();
  });

  claude.on('error', (err) => {
    broadcastEvent(proc, { type: 'error', message: err.message });
    proc.finished = true;
    for (const listener of proc.listeners) {
      listener.end();
    }
    proc.listeners.clear();
    setTimeout(() => cleanupProcess(processId), 5 * 60 * 1000);

    // Release the queue slot
    if (onFinish) onFinish();
  });

  return proc;
}

function broadcastEvent(proc, event) {
  const data = `data: ${JSON.stringify(event)}\n\n`;

  // Buffer overflow protection - write to disk if buffer gets too large
  if (proc.buffer.length >= MAX_BUFFER_EVENTS) {
    flushBufferToDisk(proc);
  }

  proc.buffer.push(event);

  for (const listener of proc.listeners) {
    try {
      listener.write(data);
    } catch (e) {
      // Listener disconnected, will be cleaned up
    }
  }
}

// Write buffer to disk and clear memory
function flushBufferToDisk(proc) {
  if (proc.buffer.length === 0) return;

  const bufferFile = path.join(BUFFER_DISK_PATH, `${proc.processId}.jsonl`);
  try {
    const lines = proc.buffer.map(e => JSON.stringify(e)).join('\n') + '\n';
    fs.appendFileSync(bufferFile, lines);
    console.log(`Flushed ${proc.buffer.length} events to disk for ${proc.processId}`);
    proc.buffer = [];
  } catch (e) {
    console.error(`Failed to flush buffer to disk: ${e.message}`);
    // Keep most recent events, drop oldest
    proc.buffer = proc.buffer.slice(-1000);
  }
}

// Load buffer from disk if it exists
function loadBufferFromDisk(processId) {
  const bufferFile = path.join(BUFFER_DISK_PATH, `${processId}.jsonl`);
  if (!fs.existsSync(bufferFile)) return [];

  try {
    const content = fs.readFileSync(bufferFile, 'utf-8');
    const events = content.split('\n')
      .filter(line => line.trim())
      .map(line => JSON.parse(line));
    console.log(`Loaded ${events.length} buffered events from disk for ${processId}`);
    return events;
  } catch (e) {
    console.error(`Failed to load buffer from disk: ${e.message}`);
    return [];
  }
}

function addListener(proc, res) {
  proc.listeners.add(res);
  res.on('close', () => {
    proc.listeners.delete(res);
    console.log(`Listener disconnected from ${proc.processId} (${proc.listeners.size} remaining)`);
  });
}

function cleanupProcess(processId) {
  const proc = activeProcesses.get(processId);
  if (!proc) return;

  // Write final snapshot to disk before cleanup
  if (proc.sessionId) {
    saveProcessSnapshot(proc);
  }

  // Clean up disk buffer
  const bufferFile = path.join(BUFFER_DISK_PATH, `${processId}.jsonl`);
  if (fs.existsSync(bufferFile)) {
    fs.unlinkSync(bufferFile);
  }

  activeProcesses.delete(processId);
  if (proc.sessionId) {
    // Only remove session mapping if it still points to this process
    if (sessionToProcess.get(proc.sessionId) === processId) {
      sessionToProcess.delete(proc.sessionId);
    }
  }
  console.log(`Cleaned up process ${processId}`);
}

// Save process state snapshot for crash recovery
function saveProcessSnapshot(proc) {
  const snapshotDir = path.join(BUFFER_DISK_PATH, 'snapshots');
  if (!fs.existsSync(snapshotDir)) {
    fs.mkdirSync(snapshotDir, { recursive: true });
  }

  const snapshot = {
    processId: proc.processId,
    sessionId: proc.sessionId,
    startTime: proc.startTime,
    finishTime: Date.now(),
    exitCode: proc.exitCode,
    finished: proc.finished,
    eventCount: proc.buffer.length,
  };

  const snapshotFile = path.join(snapshotDir, `${proc.sessionId}.json`);
  try {
    fs.writeFileSync(snapshotFile, JSON.stringify(snapshot, null, 2));
    console.log(`Saved snapshot for session ${proc.sessionId}`);
  } catch (e) {
    console.error(`Failed to save snapshot: ${e.message}`);
  }
}

// Find active process for a session
function getActiveProcess(sessionId) {
  const processId = sessionToProcess.get(sessionId);
  if (!processId) return null;
  const proc = activeProcesses.get(processId);
  if (!proc || proc.finished) return null;
  return proc;
}

// ============================================================
// Auth middleware (disabled)
// ============================================================

function authenticate(req, res, next) {
  return next();
}

// ============================================================
// Session file helpers
// ============================================================

function extractSessionTitle(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'user' && entry.message && entry.message.role === 'user') {
          const msgContent = entry.message.content;
          if (typeof msgContent === 'string') {
            return msgContent.slice(0, 80);
          }
        }
      } catch (e) {}
    }
  } catch (e) {}
  return null;
}

function parseSessionMessages(filePath) {
  const messages = [];
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'user' && entry.message && entry.message.role === 'user') {
          const msgContent = entry.message.content;
          if (typeof msgContent === 'string') {
            messages.push({ role: 'user', content: msgContent });
          }
        } else if (entry.type === 'assistant' && entry.message && entry.message.role === 'assistant') {
          const contentBlocks = entry.message.content;
          if (Array.isArray(contentBlocks)) {
            const textParts = contentBlocks
              .filter(b => b.type === 'text')
              .map(b => b.text)
              .join('');
            if (textParts) {
              const last = messages[messages.length - 1];
              if (last && last.role === 'assistant') {
                if (textParts.length > last.content.length && textParts.startsWith(last.content.slice(0, 50))) {
                  last.content = textParts;
                } else if (textParts !== last.content) {
                  messages.push({ role: 'assistant', content: textParts });
                }
              } else {
                messages.push({ role: 'assistant', content: textParts });
              }
            }
          }
        }
      } catch (e) {}
    }
  } catch (e) {}
  return messages;
}

function findSessionFile(sessionId) {
  const claudeDir = path.join(HOME_DIR, '.claude', 'projects');
  function walkDir(dir) {
    if (!fs.existsSync(dir)) return null;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const result = walkDir(fullPath);
        if (result) return result;
      } else if (entry.name === `${sessionId}.jsonl`) {
        return fullPath;
      }
    }
    return null;
  }
  return walkDir(claudeDir);
}

// ============================================================
// Sessions cache — avoid walking the filesystem on every request
// ============================================================

let sessionsCache = null;
let sessionsCacheTime = 0;
const SESSIONS_CACHE_TTL = 5000; // 5 seconds

function getSessionsList() {
  const now = Date.now();
  if (sessionsCache && (now - sessionsCacheTime) < SESSIONS_CACHE_TTL) {
    // Update running status from live data even on cache hit
    for (const s of sessionsCache) {
      s.running = !!getActiveProcess(s.id);
    }
    return sessionsCache;
  }

  const claudeDir = path.join(HOME_DIR, '.claude', 'projects');
  const sessions = [];

  try {
    function walkDir(dir) {
      if (!fs.existsSync(dir)) return;
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walkDir(fullPath);
        } else if (entry.name.endsWith('.jsonl')) {
          const stat = fs.statSync(fullPath);
          const sid = entry.name.replace('.jsonl', '');
          const title = extractSessionTitle(fullPath);
          const running = !!getActiveProcess(sid);
          sessions.push({
            id: sid,
            title: title || sid.slice(0, 8) + '...',
            modified: stat.mtime,
            size: stat.size,
            running,
          });
        }
      }
    }

    walkDir(claudeDir);
    sessions.sort((a, b) => b.modified - a.modified);
  } catch (e) {
    // return empty on error
  }

  sessionsCache = sessions.slice(0, 50);
  sessionsCacheTime = now;
  return sessionsCache;
}

// ============================================================
// Routes
// ============================================================

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// POST /api/chat — Enqueue or immediately start a Claude process
app.post('/api/chat', authenticate, (req, res) => {
  const { message, sessionId } = req.body;

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'message is required' });
  }

  // Check if session already has an active process
  if (sessionId) {
    const existing = getActiveProcess(sessionId);
    if (existing) {
      return res.status(409).json({ error: 'Session is busy processing another message' });
    }
  }

  // Set up SSE
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  enqueueOrProcess(message, sessionId, res);
});

// GET /api/process/:id/stream — Reconnect to a running process
app.get('/api/process/:id/stream', authenticate, (req, res) => {
  const processId = req.params.id;
  const proc = activeProcesses.get(processId);

  if (!proc) {
    return res.status(404).json({ error: 'Process not found' });
  }

  // Set up SSE
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  // Send disk-buffered events first (if any)
  const diskEvents = loadBufferFromDisk(processId);
  for (const event of diskEvents) {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }

  // Send memory-buffered events
  for (const event of proc.buffer) {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }

  if (proc.finished) {
    // Process already done, just send the buffer and close
    res.end();
    return;
  }

  // Subscribe to new events
  addListener(proc, res);
  console.log(`Client reconnected to process ${processId} (${diskEvents.length + proc.buffer.length} buffered events replayed)`);
});

// POST /api/process/:id/stop — Kill the Claude process
app.post('/api/process/:id/stop', authenticate, (req, res) => {
  const proc = activeProcesses.get(req.params.id);
  if (!proc) return res.status(404).json({ error: 'Process not found' });
  killProc(proc);
  res.json({ ok: true });
});

// POST /api/sessions/:id/stop — Kill by session ID (fallback when processId unknown)
app.post('/api/sessions/:id/stop', authenticate, (req, res) => {
  const proc = getActiveProcess(req.params.id);
  if (!proc) return res.status(404).json({ error: 'No active process for session' });
  killProc(proc);
  res.json({ ok: true });
});

function killProc(proc) {
  if (proc.finished) return;
  const pid = proc.claude.pid;
  console.log(`[STOP] Killing pid=${pid} (processId=${proc.processId})`);
  // SIGKILL cannot be caught or ignored — guaranteed termination
  try { process.kill(-pid, 'SIGKILL'); } catch (e) { console.log(`kill(-pgid): ${e.message}`); }
  try { process.kill(pid,  'SIGKILL'); } catch (e) { console.log(`kill(pid):   ${e.message}`); }
}

// GET /api/queue — Current queue status
app.get('/api/queue', authenticate, (req, res) => {
  res.json({
    concurrency: QUEUE_CONCURRENCY,
    active: activeCount,
    queued: messageQueue.length,
    queue: messageQueue.map((j, i) => ({
      position: i + 1,
      sessionId: j.sessionId || null,
    })),
  });
});

// GET /api/sessions — List recent sessions
app.get('/api/sessions', authenticate, (req, res) => {
  res.json(getSessionsList());
});

// GET /api/sessions/:id/messages — Get message history for a session
app.get('/api/sessions/:id/messages', authenticate, (req, res) => {
  const sessionId = req.params.id;
  const filePath = findSessionFile(sessionId);

  if (!filePath) {
    return res.status(404).json({ error: 'Session not found' });
  }

  const messages = parseSessionMessages(filePath);
  res.json(messages);
});

// DELETE /api/sessions/:id — Delete a session
app.delete('/api/sessions/:id', authenticate, (req, res) => {
  res.json({ ok: true });
});

// GET /api/health — Server and process health status
app.get('/api/health', authenticate, (req, res) => {
  const health = {
    status: 'ok',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    activeProcesses: activeProcesses.size,
    queue: { concurrency: QUEUE_CONCURRENCY, active: activeCount, queued: messageQueue.length },
    processes: [],
  };

  for (const [processId, proc] of activeProcesses) {
    const runtime = Math.floor((Date.now() - proc.startTime) / 1000);
    health.processes.push({
      processId,
      sessionId: proc.sessionId,
      finished: proc.finished,
      listeners: proc.listeners.size,
      bufferedEvents: proc.buffer.length,
      runtimeSeconds: runtime,
    });
  }

  res.json(health);
});

// GET /api/sessions/:id/snapshot — Get last known state snapshot
app.get('/api/sessions/:id/snapshot', authenticate, (req, res) => {
  const sessionId = req.params.id;
  const snapshotFile = path.join(BUFFER_DISK_PATH, 'snapshots', `${sessionId}.json`);

  if (!fs.existsSync(snapshotFile)) {
    return res.status(404).json({ error: 'No snapshot found' });
  }

  try {
    const snapshot = JSON.parse(fs.readFileSync(snapshotFile, 'utf-8'));
    res.json(snapshot);
  } catch (e) {
    res.status(500).json({ error: 'Failed to load snapshot' });
  }
});

// GET /api/usage — Get Claude usage limits from Anthropic API
app.get('/api/usage', authenticate, async (req, res) => {
  try {
    // Read OAuth token from credentials file
    const credsPath = path.join(os.homedir(), '.claude', '.credentials.json');
    const creds = JSON.parse(fs.readFileSync(credsPath, 'utf-8'));
    const accessToken = creds.claudeAiOauth?.accessToken;

    if (!accessToken) {
      return res.status(500).json({ error: 'No OAuth token found' });
    }

    // Fetch real usage data from Anthropic API
    const response = await fetch('https://api.anthropic.com/api/oauth/usage', {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'anthropic-beta': 'oauth-2025-04-20'
      }
    });

    if (!response.ok) {
      return res.status(500).json({ error: 'Failed to fetch usage data' });
    }

    const data = await response.json();
    const now = Date.now();

    // Parse reset times
    const sessionResetTime = data.five_hour?.resets_at ? new Date(data.five_hour.resets_at).getTime() : null;
    const weeklyResetTime = data.seven_day?.resets_at ? new Date(data.seven_day.resets_at).getTime() : null;

    res.json({
      usage: {
        session_used: data.five_hour?.utilization || 0,
        session_limit: 100,
        session_reset_ms: sessionResetTime ? Math.max(0, sessionResetTime - now) : 0,
        weekly_used: data.seven_day?.utilization || 0,
        weekly_limit: 100,
        weekly_reset_ms: weeklyResetTime ? Math.max(0, weeklyResetTime - now) : 0,
        extra_enabled: data.extra_usage?.is_enabled || false,
        extra_used: data.extra_usage?.used_credits || 0,
        extra_limit: data.extra_usage?.monthly_limit || 0,
        extra_utilization: data.extra_usage?.utilization || 0
      },
      timestamp: new Date().toISOString(),
      raw: data  // Include raw data for debugging
    });
  } catch (e) {
    console.error('Failed to fetch usage:', e);
    res.status(500).json({ error: e.message });
  }
});


// ============================================================
// Ollama proxy endpoints
// ============================================================

// ============================================================
// Ollama session persistence
// ============================================================

function ollamaSessionPath(sessionId) {
  return path.join(OLLAMA_SESSIONS_DIR, `${sessionId}.json`);
}

function loadOllamaSession(sessionId) {
  const p = ollamaSessionPath(sessionId);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf-8'));
  } catch (e) {
    return null;
  }
}

function saveOllamaSession(session) {
  fs.writeFileSync(ollamaSessionPath(session.id), JSON.stringify(session, null, 2));
}

function listOllamaSessions() {
  try {
    const files = fs.readdirSync(OLLAMA_SESSIONS_DIR).filter(f => f.endsWith('.json'));
    const sessions = [];
    for (const f of files) {
      try {
        const data = JSON.parse(fs.readFileSync(path.join(OLLAMA_SESSIONS_DIR, f), 'utf-8'));
        sessions.push({
          id: data.id,
          title: data.title || data.id.slice(0, 8) + '...',
          model: data.model,
          modified: data.modified,
          messageCount: (data.messages || []).filter(m => m.role !== 'system').length,
          provider: 'ollama',
        });
      } catch (e) {}
    }
    sessions.sort((a, b) => new Date(b.modified) - new Date(a.modified));
    return sessions.slice(0, 50);
  } catch (e) {
    return [];
  }
}

// GET /api/ollama/sessions — List Ollama sessions
app.get('/api/ollama/sessions', authenticate, (req, res) => {
  res.json(listOllamaSessions());
});

// GET /api/ollama/sessions/:id — Load a specific Ollama session
app.get('/api/ollama/sessions/:id', authenticate, (req, res) => {
  const session = loadOllamaSession(req.params.id);
  if (!session) return res.status(404).json({ error: 'Session not found' });
  res.json(session);
});

// DELETE /api/ollama/sessions/:id — Delete an Ollama session
app.delete('/api/ollama/sessions/:id', authenticate, (req, res) => {
  const p = ollamaSessionPath(req.params.id);
  if (fs.existsSync(p)) fs.unlinkSync(p);
  res.json({ ok: true });
});

// GET /api/ollama/models — List available Ollama models
app.get('/api/ollama/models', authenticate, async (req, res) => {
  try {
    const ollamaRes = await new Promise((resolve, reject) => {
      http.get(`${OLLAMA_BASE}/api/tags`, (r) => {
        let data = '';
        r.on('data', chunk => data += chunk);
        r.on('end', () => resolve({ status: r.statusCode, body: data }));
        r.on('error', reject);
      }).on('error', reject);
    });

    if (ollamaRes.status !== 200) {
      return res.status(502).json({ error: 'Ollama returned non-200', status: ollamaRes.status });
    }

    const parsed = JSON.parse(ollamaRes.body);
    const freeMB = await getFreeVramMB();
    const models = (parsed.models || []).map(m => {
      const vramReq = MODEL_VRAM_MB[m.name] || null;
      const fits = freeMB !== null && vramReq ? freeMB >= vramReq : null;
      return {
        name: m.name,
        size: m.size,
        modified: m.modified_at,
        family: m.details?.family || null,
        parameters: m.details?.parameter_size || null,
        vram_required_mb: vramReq,
        fits_in_vram: fits,
      };
    });
    res.json(models);
  } catch (e) {
    console.error('Failed to list Ollama models:', e.message);
    res.status(502).json({ error: `Ollama unreachable: ${e.message}` });
  }
});

// ============================================================
// Ollama tool definitions and executor
// ============================================================

const OLLAMA_TOOLS = [
  {
    type: 'function',
    function: {
      name: 'bash',
      description: 'Run a shell command and return its output. Use for system administration, checking services, reading logs, etc.',
      parameters: {
        type: 'object',
        required: ['command'],
        properties: {
          command: { type: 'string', description: 'The bash command to execute' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read the contents of a file. Returns the full text content.',
      parameters: {
        type: 'object',
        required: ['path'],
        properties: {
          path: { type: 'string', description: 'Absolute path to the file to read' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write content to a file. Creates the file if it does not exist, overwrites if it does.',
      parameters: {
        type: 'object',
        required: ['path', 'content'],
        properties: {
          path: { type: 'string', description: 'Absolute path to the file to write' },
          content: { type: 'string', description: 'The content to write' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'list_files',
      description: 'List files in a directory. Returns filenames, one per line.',
      parameters: {
        type: 'object',
        required: ['path'],
        properties: {
          path: { type: 'string', description: 'Directory path to list' },
        },
      },
    },
  },
];

const OLLAMA_MAX_ITERATIONS = 15;
const OLLAMA_CMD_TIMEOUT = 30000; // 30s per command

const OLLAMA_SYSTEM_PROMPT = `You are a helpful infrastructure assistant running on a Linux server called dev-services.
You have access to tools that let you run bash commands, read files, write files, and list directories.

When you need to investigate or fix something, USE YOUR TOOLS by calling them directly. You have sudo access (passwordless).

IMPORTANT: Do NOT write out tool calls as text/JSON in your response. Use the tool calling mechanism provided to you. Never output {"name": ...} or {"type": "function"...} as text — just call the tool directly.

Tool names available to you: bash, read_file, write_file, list_files.
- Use "bash" to run shell commands (including sudo commands)
- Use "read_file" to read file contents
- Use "write_file" to create or overwrite files
- Use "list_files" to list directory contents

Always use tools when asked about the system. Do not guess or make up information.`;

// Blocked command patterns (subset of approval-service patterns)
const OLLAMA_BLOCKED = [
  /\brm\s+-rf\s+\//,
  /\bmkfs\b/, /\bdd\s/, /\bfdisk(?!\s+-l)\b/,
  /\bsystemctl\s+(stop|disable|mask|kill)\b/,
  /\bdocker\s+(rm|rmi|kill|stop|system\s+prune)\b/,
  /\bgit\s+(reset\s+--hard|push\s+--force|clean\s+-f)\b/,
  /\bzfs\s+(destroy|rollback)\b/, /\bzpool\s+(destroy|remove)\b/,
  />\s*\/etc\//, />\s*\/boot\//, />\s*\/usr\//,
  /\bDROP\s/, /\bDELETE\s+FROM\b/, /\bTRUNCATE\s+TABLE\b/,
  /\bkillall\s/, /\bpkill\s/,
];

function isBlockedCommand(cmd) {
  return OLLAMA_BLOCKED.some(p => p.test(cmd));
}

// Parse tool calls that the model wrote as text instead of using structured tool calling
function parseTextToolCalls(text) {
  const calls = [];
  // Match JSON objects that look like tool calls: {"name": "bash", "parameters": {...}}
  const patterns = [
    /\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}/g,
    /\{\s*"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}\s*\}/g,
  ];
  const validTools = new Set(['bash', 'read_file', 'write_file', 'list_files']);

  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const name = match[1];
      if (!validTools.has(name)) continue;
      try {
        const args = JSON.parse(match[2]);
        // Normalize: "parameters" format uses "command", our tools expect that too
        calls.push({
          function: { name, arguments: args },
        });
      } catch (e) {
        // invalid JSON in args, skip
      }
    }
  }
  return calls;
}

function execTool(name, args) {
  return new Promise((resolve) => {
    switch (name) {
      case 'bash': {
        const cmd = args.command || '';
        if (isBlockedCommand(cmd)) {
          return resolve({ output: 'BLOCKED: This command is not allowed for safety reasons.', blocked: true });
        }
        const proc = spawn('bash', ['-c', cmd], {
          cwd: HOME_DIR,
          timeout: OLLAMA_CMD_TIMEOUT,
          env: { ...process.env, HOME: HOME_DIR },
        });
        let stdout = '', stderr = '';
        proc.stdout.on('data', d => stdout += d.toString());
        proc.stderr.on('data', d => stderr += d.toString());
        proc.on('close', (code) => {
          const output = (stdout + (stderr ? '\nSTDERR: ' + stderr : '')).slice(0, 8000);
          resolve({ output: output || `(exit code ${code})`, code });
        });
        proc.on('error', (err) => resolve({ output: `Error: ${err.message}`, code: 1 }));
        break;
      }
      case 'read_file': {
        const filePath = args.path || '';
        try {
          const content = fs.readFileSync(filePath, 'utf-8');
          resolve({ output: content.slice(0, 16000) });
        } catch (e) {
          resolve({ output: `Error reading file: ${e.message}` });
        }
        break;
      }
      case 'write_file': {
        const filePath = args.path || '';
        const content = args.content || '';
        // Block writes to system directories
        if (/^\/(etc|boot|usr|sys|proc|dev)\//.test(filePath)) {
          return resolve({ output: 'BLOCKED: Cannot write to system directories.', blocked: true });
        }
        try {
          fs.writeFileSync(filePath, content);
          resolve({ output: `Written ${content.length} bytes to ${filePath}` });
        } catch (e) {
          resolve({ output: `Error writing file: ${e.message}` });
        }
        break;
      }
      case 'list_files': {
        const dirPath = args.path || '.';
        try {
          const entries = fs.readdirSync(dirPath, { withFileTypes: true });
          const listing = entries.map(e => e.isDirectory() ? e.name + '/' : e.name).join('\n');
          resolve({ output: listing || '(empty directory)' });
        } catch (e) {
          resolve({ output: `Error listing directory: ${e.message}` });
        }
        break;
      }
      default:
        resolve({ output: `Unknown tool: ${name}` });
    }
  });
}

// Make a single Ollama API call (non-streaming for tool loop, streaming for final)
function ollamaRequest(payload) {
  return new Promise((resolve, reject) => {
    const url = new URL(`${OLLAMA_BASE}/api/chat`);
    const data = JSON.stringify(payload);
    const req = http.request({
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    }, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk.toString());
      res.on('end', () => {
        try {
          // Ollama non-streaming returns one JSON object
          resolve(JSON.parse(body));
        } catch (e) {
          reject(new Error(`Invalid JSON from Ollama: ${body.slice(0, 200)}`));
        }
      });
      res.on('error', reject);
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

// Stream an Ollama response, writing SSE events to res. Returns accumulated message.
function ollamaStream(payload, res) {
  return new Promise((resolve, reject) => {
    const url = new URL(`${OLLAMA_BASE}/api/chat`);
    const data = JSON.stringify({ ...payload, stream: true });
    let fullText = '';
    let toolCalls = [];
    let lastStats = null;

    const req = http.request({
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    }, (ollamaRes) => {
      let buffer = '';
      ollamaRes.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const obj = JSON.parse(line);
            if (obj.message?.content) {
              fullText += obj.message.content;
              try {
                res.write(`data: ${JSON.stringify({
                  type: 'assistant',
                  message: { content: [{ type: 'text', text: obj.message.content }] },
                  delta: true,
                })}\n\n`);
              } catch (e) {}
            }
            if (obj.message?.tool_calls) {
              toolCalls.push(...obj.message.tool_calls);
            }
            if (obj.done) {
              lastStats = {};
              if (obj.total_duration) lastStats.total_duration_ms = Math.round(obj.total_duration / 1e6);
              if (obj.eval_count) lastStats.tokens = obj.eval_count;
              if (obj.eval_duration && obj.eval_count) {
                lastStats.tokens_per_second = Math.round(obj.eval_count / (obj.eval_duration / 1e9) * 10) / 10;
              }
            }
          } catch (e) {}
        }
      });
      ollamaRes.on('end', () => {
        if (buffer.trim()) {
          try {
            const obj = JSON.parse(buffer);
            if (obj.message?.tool_calls) toolCalls.push(...obj.message.tool_calls);
            if (obj.message?.content) fullText += obj.message.content;
          } catch (e) {}
        }
        resolve({ text: fullText, tool_calls: toolCalls, stats: lastStats });
      });
      ollamaRes.on('error', reject);
    });
    req.on('error', reject);
    // Store request on res for abort on disconnect
    res._ollamaReq = req;
    req.write(data);
    req.end();
  });
}

// POST /api/ollama/chat — Tool-calling chat loop with Ollama (session-aware)
app.post('/api/ollama/chat', authenticate, async (req, res) => {
  let { model, messages, sessionId } = req.body;

  if (!model || !messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'model and messages[] are required' });
  }

  // VRAM guard — prevent gibberish from GPU memory pressure
  const vram = await vramCheck(model);
  if (!vram.ok) {
    return res.status(503).json({ error: vram.warning });
  }
  if (vram.model !== model) {
    model = vram.model;
  }

  // Load or create session
  let session;
  if (sessionId) {
    session = loadOllamaSession(sessionId);
  }
  if (!session) {
    session = {
      id: sessionId || crypto.randomUUID(),
      model,
      title: null,
      created: new Date().toISOString(),
      modified: new Date().toISOString(),
      messages: [],  // persisted messages (user + assistant text only, no system/tool noise)
    };
  }

  // Set up SSE
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  // Send session ID immediately so frontend can track it
  try {
    res.write(`data: ${JSON.stringify({ type: 'session', session_id: session.id })}\n\n`);
  } catch (e) {}

  // Notify client if model was downgraded due to VRAM pressure
  if (vram.warning) {
    try {
      res.write(`data: ${JSON.stringify({
        type: 'assistant',
        message: { content: [{ type: 'text', text: `> **${vram.warning}**\n\n` }] },
        delta: true,
      })}\n\n`);
    } catch (e) {}
  }

  let aborted = false;
  res.on('close', () => {
    aborted = true;
    if (res._ollamaReq) res._ollamaReq.destroy();
  });

  // Build full conversation for Ollama (system + all history + new user message)
  const conversation = [...messages];
  if (!conversation.some(m => m.role === 'system')) {
    conversation.unshift({ role: 'system', content: OLLAMA_SYSTEM_PROMPT });
  }

  // Extract the new user message for persistence
  const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
  if (lastUserMsg) {
    session.messages.push({ role: 'user', content: lastUserMsg.content });
    // Set title from first user message
    if (!session.title) {
      session.title = lastUserMsg.content.slice(0, 80);
    }
  }

  let totalStats = { tokens: 0, total_duration_ms: 0 };
  let finalAssistantText = '';

  try {
    for (let iter = 0; iter < OLLAMA_MAX_ITERATIONS; iter++) {
      if (aborted) break;

      // Stream the response
      const result = await ollamaStream({ model, messages: conversation, tools: OLLAMA_TOOLS }, res);

      // Accumulate stats
      if (result.stats) {
        totalStats.tokens += result.stats.tokens || 0;
        totalStats.total_duration_ms += result.stats.total_duration_ms || 0;
        if (result.stats.tokens_per_second) totalStats.tokens_per_second = result.stats.tokens_per_second;
      }

      // Track final text for session persistence
      if (result.text) finalAssistantText = result.text;

      // Check for tool calls — also parse text-based tool calls from dumber models
      if ((!result.tool_calls || result.tool_calls.length === 0) && result.text) {
        const textToolCalls = parseTextToolCalls(result.text);
        if (textToolCalls.length > 0) {
          result.tool_calls = textToolCalls;
          console.log(`[ollama-tools] Parsed ${textToolCalls.length} tool call(s) from text output`);
        }
      }

      if (!result.tool_calls || result.tool_calls.length === 0) {
        // No tools — model is done
        break;
      }

      // Add assistant message with tool calls to conversation
      const assistantMsg = { role: 'assistant' };
      if (result.text) assistantMsg.content = result.text;
      assistantMsg.tool_calls = result.tool_calls;
      conversation.push(assistantMsg);

      // Execute each tool call
      for (const tc of result.tool_calls) {
        if (aborted) break;
        const fnName = tc.function?.name;
        const fnArgs = tc.function?.arguments || {};

        console.log(`[ollama-tools] Executing: ${fnName}(${JSON.stringify(fnArgs).slice(0, 200)})`);

        // Send tool_use event to frontend
        const toolDetail = fnName === 'bash' ? (fnArgs.command || '').slice(0, 120)
          : fnName === 'read_file' ? fnArgs.path
          : fnName === 'write_file' ? `Writing ${fnArgs.path}`
          : fnName === 'list_files' ? fnArgs.path
          : '';
        try {
          res.write(`data: ${JSON.stringify({
            type: 'tool_use',
            name: fnName,
            input: fnArgs,
            detail: toolDetail,
          })}\n\n`);
        } catch (e) {}

        const toolResult = await execTool(fnName, fnArgs);

        console.log(`[ollama-tools] Result: ${toolResult.output.slice(0, 200)}${toolResult.blocked ? ' [BLOCKED]' : ''}`);

        // Send tool_result event to frontend
        try {
          res.write(`data: ${JSON.stringify({
            type: 'tool_result',
            name: fnName,
            output: toolResult.output.slice(0, 2000),
            blocked: toolResult.blocked || false,
          })}\n\n`);
        } catch (e) {}

        // Add tool result to conversation
        conversation.push({
          role: 'tool',
          content: toolResult.output,
        });
      }
    }
  } catch (e) {
    if (!aborted) {
      try {
        res.write(`data: ${JSON.stringify({ type: 'error', message: e.message })}\n\n`);
      } catch (writeErr) {}
    }
  }

  // Persist assistant response to session
  if (finalAssistantText) {
    session.messages.push({ role: 'assistant', content: finalAssistantText });
  }
  session.modified = new Date().toISOString();
  session.model = model;
  saveOllamaSession(session);

  // Send final stats and done
  if (!aborted) {
    try {
      res.write(`data: ${JSON.stringify({ type: 'result', stats: totalStats, session_id: session.id })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'done', exitCode: 0 })}\n\n`);
    } catch (e) {}
    res.end();
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Claude Chat server listening on port ${PORT}`);
  console.log(`Message queue: concurrency=${QUEUE_CONCURRENCY}`);
  console.log(`Ollama backend: ${OLLAMA_BASE}`);
});
