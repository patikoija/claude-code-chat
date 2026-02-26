const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const os = require('os');

const app = express();
const PORT = process.env.PORT || 3200;
const HOME_DIR = process.env.CLAUDE_CHAT_HOME || os.homedir();

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

const QUEUE_CONCURRENCY = parseInt(process.env.QUEUE_CONCURRENCY || '1', 10);
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


app.listen(PORT, '0.0.0.0', () => {
  console.log(`Claude Chat server listening on port ${PORT}`);
  console.log(`Message queue: concurrency=${QUEUE_CONCURRENCY}`);
});
