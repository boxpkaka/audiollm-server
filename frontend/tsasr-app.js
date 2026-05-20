/**
 * TS-ASR demo page.
 *
 * Flow:
 *   1. User records a short enrollment clip in-browser (WebAudio, 16 kHz mono).
 *   2. Clip is wrapped in a WAV container, base64-encoded, and stashed locally.
 *   3. On mic click the page opens a fresh WS to /transcribe-target-streaming
 *      and sends:
 *          { type: "start", enrollment_audio: "<b64>",
 *            enrollment_format: "wav", voice_traits?, language?, config? }
 *      If the backend replies with ``enrollment_ok`` the mic starts streaming;
 *      if it replies with an ``error`` the session is torn down and the user
 *      is prompted to re-record.
 *   4. Live PCM is sent as binary frames (Int16 mono @16 kHz) on the same WS.
 *      Final transcripts arrive as ``{type:"final", text, task:"tsasr"}``.
 */

(() => {
  'use strict';

  // TS-ASR demo page module.
  //
  // Wrapped in an ``init`` factory so the SPA router can mount and tear
  // down this page repeatedly within a single document. ``init``
  // returns a ``dispose`` callback the router calls before swapping
  // the page out — that closes the live WebSocket, releases both the
  // enrollment and live AudioContext + microphone, abort in-flight
  // uploads, revokes every cached blob URL (segment replay + enrollment
  // preview), clears the timer driving the enrollment progress bar,
  // and unsubscribes from i18n change events.
  function initTsasr() {
    const i18n = window.Amphion && window.Amphion.i18n;
    const t = (key, vars) => (i18n ? i18n.t(key, vars) : (vars && vars.defaultValue) || key);
    const onLangChange = (fn) => (i18n ? i18n.onChange(fn) : () => {});
    let i18nUnsub = null;

    const MIN_ENROLL_SEC = 1.0;
  // Backend VAD-trims longer uploads to 8s (see tsasr_enrollment_max_sec).
  // Keeping the browser auto-stop in sync avoids uploading material we know
  // will be discarded and lets the progress bar fill cleanly at 8s.
  const MAX_ENROLL_SEC = 8.0;
  const TARGET_SAMPLE_RATE = 16000;

  // -------------------- State --------------------
  let ws = null;
  let liveCtx = null;
  let liveNode = null;
  let liveStream = null;
  let isRecordingLive = false;
  // True between sending ``{type:"stop"}`` and the server closing the
  // WebSocket. The mic button is gated off during this window so the
  // user can't fire a second start before the previous take's final
  // arrives — see ``updateMicGate`` and ``ws.onclose``.
  let isAwaitingFinalize = false;

  let enrollCtx = null;
  let enrollNode = null;
  let enrollStream = null;
  let enrollChunks = []; // Float32Array pieces at 16 kHz
  let enrollStartAt = 0;
  let enrollTimerId = null;
  let enrollPcm = null; // concatenated Float32Array
  let enrollWavB64 = null;
  let enrollDurationSec = 0;
  let isEnrollRecording = false;
  let enrollPreviewUrl = null;

  // Track current displayed enrollment status so we can re-render on lang switch.
  let enrollStatusDyn = { state: 'idle', key: 'tsasr.enroll.notRecorded', vars: null };

  // -------------------- Segment replay cache --------------------
  // Keyed by the backend-assigned `final.id`. Each value is a blob URL that
  // can be handed to an <audio> element. We eagerly create the blob on
  // message arrival so the replay button responds instantly, then revoke on
  // reset or page unload to avoid leaking audio blobs.
  const segmentAudio = new Map();
  let activeReplayAudio = null;

  // Partial bubbles still waiting for their `final` counterpart. Keyed by
  // utterance id (same id the backend reuses for the eventual final), so
  // that when the final arrives we can replace the partial's text in
  // place — no second `chat-bubble-float` animation, no extra DOM row.
  const partialBubbles = new Map();

  function b64ToWavBlobUrl(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  }

  function clearSegmentAudio() {
    if (activeReplayAudio) {
      try { activeReplayAudio.pause(); } catch { /* noop */ }
      activeReplayAudio = null;
    }
    segmentAudio.forEach((url) => {
      try { URL.revokeObjectURL(url); } catch { /* noop */ }
    });
    segmentAudio.clear();
    partialBubbles.clear();
    document.querySelectorAll('.replay-btn.is-playing').forEach((b) => {
      b.classList.remove('is-playing');
    });
  }

  // beforeunload still fires on real tab close. We use ``onBeforeUnload``
  // so dispose can detach the listener — otherwise multiple SPA mounts
  // would stack up duplicate listeners on the window.
  function onBeforeUnload() { clearSegmentAudio(); }
  window.addEventListener('beforeunload', onBeforeUnload);

  // -------------------- DOM refs --------------------
  const micBtn = document.getElementById('mic-btn');
  const micIcon = document.getElementById('mic-icon');
  const micStatus = document.getElementById('mic-status');
  const pulseRings = document.querySelectorAll('.pulse-ring');
  const chatArea = document.getElementById('chat-area');

  const enrollStatusPill = document.getElementById('enroll-status');
  const enrollRecBtn = document.getElementById('enroll-rec-btn');
  const enrollRecLabel = document.getElementById('enroll-rec-label');
  const enrollTimer = document.getElementById('enroll-timer');
  const enrollProgressBar = document.getElementById('enroll-progress-bar');
  const enrollProgress = document.getElementById('enroll-progress');
  const enrollPreviewEl = document.getElementById('enroll-preview');

  const uploadBtn = document.getElementById('upload-btn');
  const uploadBtnLabel = uploadBtn ? uploadBtn.querySelector('.btn-upload-label') : null;
  const uploadInput = document.getElementById('upload-input');
  const uploadStatus = document.getElementById('upload-status');

  const enrollUploadBtn = document.getElementById('enroll-upload-btn');
  const enrollUploadBtnLabel = enrollUploadBtn
    ? enrollUploadBtn.querySelector('.btn-upload-label')
    : null;
  const enrollUploadInput = document.getElementById('enroll-upload-input');
  const enrollUploadStatus = document.getElementById('enroll-upload-status');
  let isEnrollUploading = false;
  let currentEnrollUploadDyn = null; // { key, vars }

  // Upload state for the transcription stage. The transcription upload now
  // hits POST /api/tsasr/upload as a one-shot REST call (mixed audio in
  // multipart, enrollment WAV inlined as base64) and never opens a WS.
  let isUploading = false;
  let uploadController = null;     // AbortController for in-flight fetch
  let currentUploadDyn = null;     // { key, vars }
  const TSASR_UPLOAD_MAX_SECONDS = 60;

  // -------------------- Hotword state --------------------
  // Mirrors the realtime-ASR page's hotword pipeline (manual entry +
  // long-text LLM extraction) with TS-ASR-specific persistence keys
  // and DOM IDs (``tsasr-hotword-*``) so the two pages can keep
  // independent hotword lists. The hotword list is comma-joined and
  // sent on every ``start`` (initial state) and ``update_hotwords``
  // (mid-session edit) message — see the v3 SFT prompt template B in
  // ``backend/tsasr/prompt.py`` for the receiving end.
  const TSASR_HOTWORDS_KEY = 'tsasr_hotwords';
  const TSASR_HOTWORD_ENABLED_KEY = 'tsasr_hotword_enabled';
  const TSASR_HOTWORD_MAX = 100;
  const TSASR_EXTRACTED_HOTWORD_MAX_LEN = 10;

  let hotwords = [];
  let hotwordEnabled = localStorage.getItem(TSASR_HOTWORD_ENABLED_KEY) !== '0';
  let extractRequestId = null;
  let currentExtractDyn = { key: 'tsasr.extract.idle', vars: null };

  const hotwordInput = document.getElementById('tsasr-hotword-input');
  const hotwordAddBtn = document.getElementById('tsasr-hotword-add-btn');
  const hotwordList = document.getElementById('tsasr-hotword-list');
  const hotwordClearBtn = document.getElementById('tsasr-hotword-clear-btn');
  const hotwordEnabledInput = document.getElementById('tsasr-hotword-enabled');
  const hotwordSyncStatus = document.getElementById('tsasr-hotword-sync-status');
  const hotwordCount = document.getElementById('tsasr-hotword-count');
  const hotwordTextarea = document.getElementById('tsasr-hotword-textarea');
  const hotwordExtractBtn = document.getElementById('tsasr-hotword-extract-btn');
  const hotwordExtractStatus = document.getElementById('tsasr-hotword-extract-status');

  function langDisplayName(value) {
    if (!value) return '';
    const v = String(value).trim();
    if (!v) return '';
    return t(`lang.name.${v}`, { defaultValue: v });
  }

  // -------------------- Connection status --------------------
  function setConnStatus(state) {
    if (!window.AmphionSidebar || !window.AmphionSidebar.setConnectionState) return;
    if (state === 'connected') {
      window.AmphionSidebar.setConnectionState('connected');
    } else if (state === 'pending') {
      window.AmphionSidebar.setConnectionState('pending');
    } else {
      window.AmphionSidebar.setConnectionState('idle');
    }
  }
  setConnStatus('disconnected');

  // -------------------- Hotword management --------------------
  function setDynText(el, key, vars) {
    if (!el) return;
    el.setAttribute('data-dyn-key', key);
    if (vars) {
      try {
        el.setAttribute('data-dyn-vars', JSON.stringify(vars));
      } catch (_) {
        el.removeAttribute('data-dyn-vars');
      }
    } else {
      el.removeAttribute('data-dyn-vars');
    }
    el.textContent = t(key, vars || undefined);
  }

  function sanitizeHotwords(sourceWords) {
    const seen = new Set();
    const cleaned = [];
    (Array.isArray(sourceWords) ? sourceWords : []).forEach((w) => {
      const s = String(w || '').trim();
      if (!s || seen.has(s)) return;
      seen.add(s);
      cleaned.push(s);
    });
    return cleaned.slice(0, TSASR_HOTWORD_MAX);
  }

  function readHotwordsFromStorage() {
    const raw = localStorage.getItem(TSASR_HOTWORDS_KEY);
    if (!raw) return [];
    try {
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? sanitizeHotwords(arr) : [];
    } catch {
      return [];
    }
  }

  function persistHotwords() {
    localStorage.setItem(TSASR_HOTWORDS_KEY, JSON.stringify(hotwords));
  }

  function getEffectiveHotwords() {
    return hotwordEnabled ? hotwords.slice() : [];
  }

  function renderHotwords() {
    if (!hotwordList) return;
    hotwordList.innerHTML = '';
    hotwords.forEach((word, idx) => {
      const tag = document.createElement('span');
      tag.className = 'hotword-pill';
      tag.innerHTML =
        `${escapeHtml(word)}` +
        `<button data-idx="${idx}" aria-label="${escapeHtml(t('tsasr.hotword.removeAria'))}">&times;</button>`;
      tag.querySelector('button').addEventListener('click', () => removeHotword(idx));
      hotwordList.appendChild(tag);
    });
    if (hotwordCount) {
      setDynText(hotwordCount, 'tsasr.hotword.count', { n: hotwords.length });
    }
  }

  const SYNC_PILL_BASE = 'status-pill';
  let currentSyncState = 'waiting';

  function setHotwordSyncStatus(state) {
    if (!hotwordSyncStatus) return;
    currentSyncState = state;
    hotwordSyncStatus.className = SYNC_PILL_BASE;
    if (state === 'synced') {
      const key = hotwordEnabled ? 'tsasr.sync.active' : 'tsasr.sync.paused';
      setDynText(hotwordSyncStatus, key);
      hotwordSyncStatus.dataset.state = hotwordEnabled ? 'ready' : 'waiting';
      return;
    }
    if (state === 'offline') {
      setDynText(hotwordSyncStatus, 'tsasr.sync.offline');
      hotwordSyncStatus.dataset.state = 'offline';
      return;
    }
    setDynText(hotwordSyncStatus, 'tsasr.sync.waiting');
    hotwordSyncStatus.dataset.state = 'waiting';
  }

  function syncHotwords() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(
          JSON.stringify({
            type: 'update_hotwords',
            hotwords: getEffectiveHotwords(),
          })
        );
      } catch (_) {
        setHotwordSyncStatus('offline');
        return;
      }
      setHotwordSyncStatus('synced');
    } else {
      setHotwordSyncStatus('offline');
    }
  }

  function saveAndSyncHotwords() {
    hotwords = sanitizeHotwords(hotwords);
    persistHotwords();
    renderHotwords();
    syncHotwords();
  }

  function setExtractStatus(state, key, vars) {
    if (!hotwordExtractStatus) return;
    currentExtractDyn = { key, vars: vars || null };
    setDynText(hotwordExtractStatus, key, vars || undefined);
    hotwordExtractStatus.className = 'hotword-extract-status';
    if (state === 'loading') hotwordExtractStatus.classList.add('is-loading');
    else if (state === 'success') hotwordExtractStatus.classList.add('is-success');
    else if (state === 'error') hotwordExtractStatus.classList.add('is-error');
  }

  function setExtractBusy(busy) {
    if (!hotwordExtractBtn || !hotwordTextarea) return;
    hotwordExtractBtn.disabled = busy;
    setDynText(
      hotwordExtractBtn,
      busy ? 'tsasr.hotword.extracting' : 'tsasr.hotword.extract'
    );
    hotwordTextarea.disabled = busy;
    updateExtractButtonAttention();
  }

  function updateExtractButtonAttention() {
    if (!hotwordExtractBtn || !hotwordTextarea) return;
    const hasText = hotwordTextarea.value.trim().length > 0;
    hotwordExtractBtn.classList.toggle(
      'btn-primary-attention',
      hasText && !hotwordExtractBtn.disabled
    );
  }

  function mergeExtractedHotwords(words) {
    const incoming = sanitizeHotwords(
      (Array.isArray(words) ? words : [])
        .filter((w) => w && w.length < TSASR_EXTRACTED_HOTWORD_MAX_LEN)
    );
    let added = 0;
    incoming.forEach((word) => {
      if (!hotwords.includes(word)) {
        hotwords.push(word);
        added += 1;
      }
    });
    if (added > 0) {
      saveAndSyncHotwords();
    } else {
      renderHotwords();
    }
    return added;
  }

  function requestHotwordExtraction(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setExtractStatus('error', 'tsasr.extract.wsOffline');
      return;
    }
    const trimmed = String(text || '').trim();
    if (!trimmed) {
      setExtractStatus('error', 'tsasr.extract.pasteFirst');
      return;
    }
    if (extractRequestId) {
      setExtractStatus('error', 'tsasr.extract.alreadyRunning');
      return;
    }
    extractRequestId = `tsasr-extract-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    setExtractBusy(true);
    setExtractStatus('loading', 'tsasr.extract.loading');
    try {
      ws.send(
        JSON.stringify({
          type: 'extract_hotwords',
          request_id: extractRequestId,
          text: trimmed,
        })
      );
    } catch (_) {
      extractRequestId = null;
      setExtractBusy(false);
      setExtractStatus('error', 'tsasr.extract.wsOffline');
    }
  }

  function addHotword(text) {
    const words = String(text || '')
      .split(/[,，\n;；]+/)
      .map((s) => s.trim())
      .filter((w) => w && !hotwords.includes(w));
    if (words.length === 0) return;
    hotwords.push(...words);
    saveAndSyncHotwords();
  }

  function removeHotword(idx) {
    if (idx < 0 || idx >= hotwords.length) return;
    hotwords.splice(idx, 1);
    saveAndSyncHotwords();
  }

  function clearHotwords() {
    if (hotwords.length === 0) return;
    hotwords = [];
    saveAndSyncHotwords();
  }

  if (hotwordAddBtn && hotwordInput) {
    hotwordAddBtn.addEventListener('click', () => {
      addHotword(hotwordInput.value);
      hotwordInput.value = '';
    });
    hotwordInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        addHotword(hotwordInput.value);
        hotwordInput.value = '';
      }
    });
  }
  if (hotwordClearBtn) hotwordClearBtn.addEventListener('click', clearHotwords);
  if (hotwordExtractBtn && hotwordTextarea) {
    hotwordExtractBtn.addEventListener('click', () => {
      requestHotwordExtraction(hotwordTextarea.value);
    });
    hotwordTextarea.addEventListener('input', updateExtractButtonAttention);
  }

  if (hotwordEnabledInput) {
    hotwordEnabledInput.checked = hotwordEnabled;
    hotwordEnabledInput.addEventListener('change', () => {
      hotwordEnabled = hotwordEnabledInput.checked;
      localStorage.setItem(TSASR_HOTWORD_ENABLED_KEY, hotwordEnabled ? '1' : '0');
      syncHotwords();
    });
  }

  hotwords = readHotwordsFromStorage();
  renderHotwords();
  setHotwordSyncStatus('waiting');
  setExtractStatus('idle', 'tsasr.extract.idle');
  updateExtractButtonAttention();

  // -------------------- Enrollment status pill --------------------
  function setEnrollStatus(state, key, vars) {
    enrollStatusPill.className = 'status-pill';
    if (state === 'recording') {
      enrollStatusPill.dataset.state = 'recording';
    } else if (state === 'ready') {
      enrollStatusPill.dataset.state = 'ready';
    } else if (state === 'error') {
      enrollStatusPill.dataset.state = 'error';
    } else if (state === 'pending') {
      enrollStatusPill.dataset.state = 'pending';
    } else {
      enrollStatusPill.dataset.state = 'idle';
    }
    enrollStatusDyn = { state, key, vars: vars || null };
    enrollStatusPill.textContent = t(key, vars || undefined);
  }

  function updateMicGate(messageKey) {
    const enabled =
      enrollWavB64 !== null
      && !isEnrollRecording
      && !isUploading
      && !isEnrollUploading
      // Block the mic between sending ``stop`` and the WS actually
      // closing — otherwise the user could click again, fall through
      // ``stopLiveStreaming``'s early return, and end up wondering
      // why nothing happened.
      && !isAwaitingFinalize;
    micBtn.disabled = !enabled;
    if (uploadBtn) {
      uploadBtn.disabled =
        enrollWavB64 === null
        || isEnrollRecording
        || isRecordingLive
        || isUploading
        || isEnrollUploading
        || isAwaitingFinalize;
    }
    if (enrollUploadBtn) {
      enrollUploadBtn.disabled =
        isEnrollRecording || isRecordingLive || isEnrollUploading;
    }
    if (isRecordingLive) {
      micStatus.textContent = t('tsasr.mic.listening');
      micStatus.setAttribute('data-dyn-key', 'tsasr.mic.listening');
    } else if (isAwaitingFinalize) {
      micStatus.textContent = t('tsasr.recognizing');
      micStatus.setAttribute('data-dyn-key', 'tsasr.recognizing');
    } else if (!enabled) {
      const k = messageKey || 'tsasr.mic.gateDisabled';
      micStatus.textContent = t(k);
      micStatus.setAttribute('data-dyn-key', k);
    } else {
      micStatus.textContent = t('tsasr.mic.start');
      micStatus.setAttribute('data-dyn-key', 'tsasr.mic.start');
    }
  }

  // -------------------- WAV encoder (Float32 mono @16k -> WAV bytes) --------
  function floatToPcm16(samples) {
    const out = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  function encodeWav(floatSamples, sampleRate = TARGET_SAMPLE_RATE) {
    const pcm16 = floatToPcm16(floatSamples);
    const byteLength = pcm16.length * 2;
    const buffer = new ArrayBuffer(44 + byteLength);
    const view = new DataView(buffer);
    const writeStr = (off, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + byteLength, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, byteLength, true);
    new Int16Array(buffer, 44).set(pcm16);
    return new Uint8Array(buffer);
  }

  function bytesToBase64(bytes) {
    // Chunked to avoid call-stack limits on large buffers.
    const CHUNK = 0x8000;
    let binary = '';
    for (let i = 0; i < bytes.length; i += CHUNK) {
      binary += String.fromCharCode.apply(
        null, bytes.subarray(i, i + CHUNK)
      );
    }
    return btoa(binary);
  }

  // -------------------- Audio context setup --------------------
  async function openSixteenKContext() {
    // Some browsers can't honor 16 kHz (e.g. Safari), but Chrome/Firefox do.
    // If the request isn't honored we still proceed, and the session will
    // send at whatever the browser returns — the backend already expects
    // 16 kHz so we'll warn loudly in that case.
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: { ideal: TARGET_SAMPLE_RATE },
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    if (ctx.sampleRate !== TARGET_SAMPLE_RATE) {
      console.warn(
        `AudioContext honored ${ctx.sampleRate} Hz instead of ${TARGET_SAMPLE_RATE} Hz; ` +
        'audio will be uploaded at that rate.'
      );
    }
    await ctx.audioWorklet.addModule('tsasr-processor.js?v=' + Date.now());
    const source = ctx.createMediaStreamSource(mediaStream);
    const node = new AudioWorkletNode(ctx, 'tsasr-capture-processor');
    source.connect(node);
    node.connect(ctx.destination);
    return { ctx, node, mediaStream };
  }

  // -------------------- Enrollment recording --------------------
  async function startEnrollRecording() {
    // Always start from a clean slate: clicking "Start recording" discards
    // any previous enrollment (audio buffers, preview, in-flight uploads,
    // status pill) so the user can re-record without a separate Reset btn.
    discardPreviousEnrollment();

    try {
      const { ctx, node, mediaStream } = await openSixteenKContext();
      enrollCtx = ctx;
      enrollNode = node;
      enrollStream = mediaStream;
    } catch (err) {
      console.error(err);
      setEnrollStatus('error', 'tsasr.enroll.micDenied');
      alert(t('tsasr.enroll.micAlert'));
      return;
    }

    enrollChunks = [];
    enrollNode.port.onmessage = (evt) => {
      if (evt.data.type === 'audio') {
        enrollChunks.push(evt.data.samples);
      }
    };
    isEnrollRecording = true;
    enrollStartAt = performance.now();
    enrollRecLabel.textContent = t('tsasr.enroll.stop');
    enrollRecLabel.setAttribute('data-i18n', 'tsasr.enroll.stop');
    enrollRecBtn.classList.add('enroll-recording');
    setEnrollStatus('recording', 'tsasr.enroll.recording');
    enrollPreviewEl.classList.add('hidden');

    enrollTimerId = setInterval(tickEnrollTimer, 80);
  }

  function tickEnrollTimer() {
    const dt = (performance.now() - enrollStartAt) / 1000;
    enrollTimer.textContent = `${dt.toFixed(1)}s`;
    const pct = Math.min(100, (dt / MAX_ENROLL_SEC) * 100);
    enrollProgressBar.style.width = `${pct}%`;
    if (dt >= MAX_ENROLL_SEC) {
      stopEnrollRecording();
    }
  }

  async function stopEnrollRecording() {
    if (!isEnrollRecording) return;
    isEnrollRecording = false;
    clearInterval(enrollTimerId);
    enrollTimerId = null;
    enrollRecBtn.classList.remove('enroll-recording');
    enrollRecLabel.textContent = t('tsasr.enroll.start');
    enrollRecLabel.setAttribute('data-i18n', 'tsasr.enroll.start');

    if (enrollNode) {
      enrollNode.port.onmessage = null;
      enrollNode.disconnect();
      enrollNode = null;
    }
    if (enrollCtx) {
      await enrollCtx.close();
      enrollCtx = null;
    }
    if (enrollStream) {
      enrollStream.getTracks().forEach((tr) => tr.stop());
      enrollStream = null;
    }

    const total = enrollChunks.reduce((n, b) => n + b.length, 0);
    const sr = TARGET_SAMPLE_RATE;
    const duration = total / sr;
    enrollDurationSec = duration;
    enrollTimer.textContent = `${duration.toFixed(1)}s`;

    if (duration < MIN_ENROLL_SEC) {
      setEnrollStatus('error', 'tsasr.enroll.tooShort', { dur: duration.toFixed(1) });
      enrollChunks = [];
      enrollPcm = null;
      enrollWavB64 = null;
      updateMicGate();
      return;
    }

    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of enrollChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    enrollPcm = merged;
    const wavBytes = encodeWav(merged, sr);
    enrollWavB64 = bytesToBase64(wavBytes);

    // Preview
    if (enrollPreviewUrl) URL.revokeObjectURL(enrollPreviewUrl);
    enrollPreviewUrl = URL.createObjectURL(
      new Blob([wavBytes], { type: 'audio/wav' })
    );
    enrollPreviewEl.src = enrollPreviewUrl;
    enrollPreviewEl.classList.remove('hidden');

    setEnrollStatus('ready', 'tsasr.enroll.ready', { dur: duration.toFixed(1) });
    updateMicGate();
  }

  // Drop any previously captured enrollment without flipping button state
  // for a separate Reset affordance — invoked at the start of each new
  // recording so the user gets a clean buffer + UI.
  function discardPreviousEnrollment() {
    enrollChunks = [];
    enrollPcm = null;
    enrollWavB64 = null;
    enrollDurationSec = 0;
    enrollProgressBar.style.width = '0%';
    enrollTimer.textContent = '0.0s';
    enrollPreviewEl.classList.add('hidden');
    if (enrollPreviewUrl) {
      URL.revokeObjectURL(enrollPreviewUrl);
      enrollPreviewUrl = null;
    }
    setEnrollStatus('idle', 'tsasr.enroll.notRecorded');
    if (uploadController) {
      try { uploadController.abort(); } catch (_) { /* noop */ }
      uploadController = null;
    }
    setUploadStatus(null, null);
    setEnrollUploadStatus(null, null);
    updateMicGate();
  }

  enrollRecBtn.addEventListener('click', () => {
    if (isRecordingLive) return; // ignore while streaming
    if (isEnrollRecording) {
      stopEnrollRecording();
    } else {
      startEnrollRecording();
    }
  });

  // -------------------- Enrollment via uploaded audio --------------------
  // Mirrors stopEnrollRecording's tail-end work (truncate, encode WAV,
  // populate preview + enrollWavB64) but the source PCM comes from a
  // user-picked file decoded by AmphionAudioUpload instead of the mic.

  function setEnrollUploadStatus(state, key, vars) {
    if (!enrollUploadStatus) return;
    if (!key) {
      enrollUploadStatus.hidden = true;
      enrollUploadStatus.textContent = '';
      enrollUploadStatus.removeAttribute('data-state');
      currentEnrollUploadDyn = null;
      return;
    }
    enrollUploadStatus.hidden = false;
    enrollUploadStatus.dataset.state = state || 'info';
    currentEnrollUploadDyn = { key, vars: vars || null };
    enrollUploadStatus.textContent = t(key, vars || undefined);
  }

  function setEnrollUploadBusy(busy) {
    isEnrollUploading = busy;
    if (enrollUploadBtn) {
      enrollUploadBtn.disabled = busy || isEnrollRecording || isRecordingLive;
    }
    if (enrollUploadBtnLabel) {
      enrollUploadBtnLabel.textContent = t(
        busy ? 'tsasr.enrollUpload.uploading' : 'tsasr.enrollUpload.label'
      );
    }
    // While uploading enrollment, also gate the mic / file-upload for the
    // transcription stage so the user can't fire two flows at once.
    enrollRecBtn.disabled = busy;
    updateMicGate();
  }

  async function handleEnrollUploadFile(file) {
    if (!file) return;
    if (isEnrollRecording || isRecordingLive || isEnrollUploading || isUploading) {
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.busy');
      return;
    }
    const upload = window.AmphionAudioUpload;
    if (!upload) {
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.unsupported');
      return;
    }

    setEnrollUploadBusy(true);
    setEnrollUploadStatus('info', 'tsasr.enrollUpload.decoding');

    let pcm;
    try {
      pcm = await upload.decodeFileToMono(file, TARGET_SAMPLE_RATE);
    } catch (err) {
      console.error('Enrollment upload decode failed:', err);
      setEnrollUploadBusy(false);
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.decode');
      return;
    }
    if (!pcm || pcm.length === 0) {
      setEnrollUploadBusy(false);
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.empty');
      return;
    }

    const sr = TARGET_SAMPLE_RATE;
    const totalSec = pcm.length / sr;
    if (totalSec < MIN_ENROLL_SEC) {
      setEnrollUploadBusy(false);
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.tooShort', {
        dur: totalSec.toFixed(1),
        min: MIN_ENROLL_SEC.toFixed(1),
      });
      return;
    }

    let trimmedNote = null;
    if (totalSec > MAX_ENROLL_SEC) {
      // Match the live recorder's auto-stop behavior: keep the leading
      // MAX_ENROLL_SEC seconds, drop the rest. The backend VAD-trims
      // anyway, but trimming up-front gives a tidy preview waveform.
      pcm = new Float32Array(pcm.subarray(0, Math.floor(sr * MAX_ENROLL_SEC)));
      trimmedNote = totalSec.toFixed(1);
    }
    const finalDuration = pcm.length / sr;

    // Update the recorder UI state so the existing "ready" affordances
    // (preview, reset button, status pill, gating) light up as if the
    // enrollment had been recorded live.
    enrollPcm = pcm;
    enrollDurationSec = finalDuration;
    enrollTimer.textContent = `${finalDuration.toFixed(1)}s`;
    enrollProgressBar.style.width = `${Math.min(100, (finalDuration / MAX_ENROLL_SEC) * 100)}%`;

    const wavBytes = encodeWav(pcm, sr);
    enrollWavB64 = bytesToBase64(wavBytes);

    if (enrollPreviewUrl) URL.revokeObjectURL(enrollPreviewUrl);
    enrollPreviewUrl = URL.createObjectURL(
      new Blob([wavBytes], { type: 'audio/wav' })
    );
    enrollPreviewEl.src = enrollPreviewUrl;
    enrollPreviewEl.classList.remove('hidden');

    setEnrollStatus('ready', 'tsasr.enroll.ready', { dur: finalDuration.toFixed(1) });
    setEnrollUploadBusy(false);
    if (trimmedNote !== null) {
      setEnrollUploadStatus('warn', 'tsasr.enrollUpload.trimmed', {
        max: MAX_ENROLL_SEC.toFixed(1),
        actual: trimmedNote,
      });
    } else {
      setEnrollUploadStatus('success', 'tsasr.enrollUpload.done', {
        dur: finalDuration.toFixed(1),
      });
    }
    updateMicGate();
  }

  if (enrollUploadBtn && enrollUploadInput) {
    enrollUploadBtn.addEventListener('click', () => {
      if (enrollUploadBtn.disabled) return;
      enrollUploadInput.value = '';
      enrollUploadInput.click();
    });
    enrollUploadInput.addEventListener('change', () => {
      const file = enrollUploadInput.files && enrollUploadInput.files[0];
      if (file) handleEnrollUploadFile(file);
    });
  }

  // -------------------- Transcript UI --------------------
  function replaySegment(segId, btn) {
    // Toggle behavior: clicking the playing button (or any button while
    // something is playing) stops the current audio. Clicking a different
    // segment's button starts a fresh playback.
    if (activeReplayAudio) {
      activeReplayAudio.pause();
      const prevBtn = document.querySelector('.replay-btn.is-playing');
      if (prevBtn) prevBtn.classList.remove('is-playing');
      const wasSame = activeReplayAudio._segId === segId;
      activeReplayAudio = null;
      if (wasSame) return;
    }
    const url = segmentAudio.get(segId);
    if (!url) return;
    const audio = new Audio(url);
    audio._segId = segId;
    if (btn) btn.classList.add('is-playing');
    audio.addEventListener('ended', () => {
      if (btn) btn.classList.remove('is-playing');
      if (activeReplayAudio === audio) activeReplayAudio = null;
    });
    audio.play().catch(() => {
      if (btn) btn.classList.remove('is-playing');
    });
    activeReplayAudio = audio;
  }

  function buildBubbleSkeleton(segId, isPartial) {
    const wrapper = document.createElement('div');
    // Apply chat-bubble-float only on the very first DOM insertion so the
    // fly-in animation runs once at the start of pseudo-streaming and the
    // partial -> final transition stays smooth (no second animation).
    wrapper.className = 'chat-row chat-row-ai chat-bubble-float';
    if (segId) wrapper.id = `ai-${segId}`;
    if (isPartial) wrapper.dataset.partial = '1';
    // Dual-ASR layout: two labeled rows ("安菲翁:" + "千问:") side-by-side
    // with the same text-frame helper driving each one independently. The
    // replay slot sits at the top-right so it stays put across partial /
    // final / processing states.
    const labelPrimary = escapeHtml(t('tsasr.label.primary'));
    const labelSecondary = escapeHtml(t('tsasr.label.secondary'));
    wrapper.innerHTML = `
      <div class="flex gap-3 max-w-2xl items-start">
        <div class="chat-avatar flex-shrink-0">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
          </svg>
        </div>
        <div class="chat-bubble chat-bubble-ai ai-content">
          <div class="flex items-start gap-2">
            <div class="flex-1 dual-asr-lines">
              <div class="bubble-line bubble-line-primary">
                <span class="bubble-label" data-dyn-key="tsasr.label.primary">${labelPrimary}</span>
                <p class="text-sm leading-relaxed bubble-text bubble-text-primary"></p>
              </div>
              <div class="bubble-line bubble-line-secondary">
                <span class="bubble-label" data-dyn-key="tsasr.label.secondary">${labelSecondary}</span>
                <p class="text-sm leading-relaxed bubble-text bubble-text-secondary"></p>
              </div>
            </div>
            <span class="bubble-replay-slot"></span>
          </div>
          <div class="bubble-meta-slot"></div>
        </div>
      </div>
    `;
    return wrapper;
  }

  function applyMeta(wrapper, langValue, durationSec) {
    const slot = wrapper.querySelector('.bubble-meta-slot');
    if (!slot) return;
    const metaParts = [];
    if (langValue) {
      metaParts.push(
        `<span data-dyn-key="tsasr.meta.lang"
               data-dyn-vars='${escapeHtml(JSON.stringify({ lang: langValue }))}'>${escapeHtml(t('tsasr.meta.lang', { lang: langDisplayName(langValue) }))}</span>`
      );
    }
    if (typeof durationSec === 'number') {
      metaParts.push(`<span>${durationSec.toFixed(1)}s</span>`);
    }
    if (metaParts.length === 0) {
      slot.outerHTML = '<div class="bubble-meta-slot"></div>';
      return;
    }
    slot.outerHTML = `<div class="bubble-meta-slot text-[11px] text-faint mt-1">${metaParts.join(' \u00b7 ')}</div>`;
  }

  function applyReplayButton(wrapper, segId) {
    const slot = wrapper.querySelector('.bubble-replay-slot');
    if (!slot) return;
    if (!segId || !segmentAudio.has(segId)) {
      slot.outerHTML = '<span class="bubble-replay-slot"></span>';
      return;
    }
    const replayTitle = escapeHtml(t('tsasr.replayTitle'));
    const btnHtml = `<button class="replay-btn bubble-replay-slot" type="button" title="${replayTitle}">
        <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
          <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
        </svg>
      </button>`;
    slot.outerHTML = btnHtml;
    const btn = wrapper.querySelector('button.bubble-replay-slot');
    if (btn) {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        replaySegment(segId, e.currentTarget);
      });
    }
  }

  // Route every text mutation through the streaming-text helper so the
  // partial -> partial and partial -> final transitions only animate the
  // characters that actually changed. Falls back to plain textContent if
  // the helper script isn't loaded for some reason (defensive — both
  // pages list it in the script tag chain).
  function setBubbleText(textEl, text) {
    if (!textEl) return;
    const next = text == null ? '' : String(text);
    if (window.AmphionStreamingText && window.AmphionStreamingText.apply) {
      window.AmphionStreamingText.apply(textEl, next);
    } else {
      textEl.textContent = next;
    }
  }

  // Apply text to the dual-ASR rows. Each row is updated independently
  // through the streaming-text helper so the partial -> partial and
  // partial -> final transitions only crossfade the row that actually
  // changed (the other row stays put if its text was already current).
  function setDualBubbleText(wrapper, textPrimary, textSecondary) {
    if (!wrapper) return;
    setBubbleText(
      wrapper.querySelector('.bubble-text-primary'),
      textPrimary == null ? '' : textPrimary,
    );
    setBubbleText(
      wrapper.querySelector('.bubble-text-secondary'),
      textSecondary == null ? '' : textSecondary,
    );
  }

  function addPartialBubble(segId, text, textSecondary) {
    const wrapper = buildBubbleSkeleton(segId, true);
    setDualBubbleText(wrapper, text, textSecondary);
    chatArea.appendChild(wrapper);
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    if (segId) partialBubbles.set(segId, wrapper);
  }

  // Placeholder bubble shown while the model is running. Reuses the
  // partial-bubble book-keeping (``partialBubbles`` map + the same
  // skeleton DOM) so ``addFinalBubble`` can upgrade it in place once the
  // final transcript arrives. The ``is-recognizing`` class drives a
  // slow opacity / shimmer breath in CSS.
  //
  // For the dual-ASR layout we paint the breathing "识别中…" placeholder
  // on the primary (Amphion) row only — having both rows pulse at the
  // same time double-stacks the animation and reads as visual noise.
  // The secondary row is left empty until its real Qwen3 text arrives.
  function addProcessingBubble(segId) {
    if (!segId) return;
    if (partialBubbles.has(segId)) return;
    const wrapper = buildBubbleSkeleton(segId, true);
    wrapper.classList.add('is-recognizing');
    const textEl = wrapper.querySelector('.bubble-text-primary');
    if (textEl) {
      textEl.classList.add('bubble-text-recognizing');
      // ``data-dyn-key`` lets refreshDynamic re-translate the
      // placeholder if the user switches language while the model is
      // still working on this segment.
      textEl.setAttribute('data-dyn-key', 'tsasr.recognizing');
      textEl.textContent = t('tsasr.recognizing');
    }
    chatArea.appendChild(wrapper);
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    partialBubbles.set(segId, wrapper);
  }

  function clearRecognizingDecor(wrapper) {
    if (!wrapper) return;
    wrapper.classList.remove('is-recognizing');
    // Walk both text rows: the placeholder is currently only painted on
    // primary, but a defensive sweep here keeps us robust if future
    // changes ever paint a secondary placeholder too.
    wrapper
      .querySelectorAll('.bubble-text-primary, .bubble-text-secondary')
      .forEach((textEl) => {
        textEl.classList.remove('bubble-text-recognizing');
        if (textEl.getAttribute('data-dyn-key') === 'tsasr.recognizing') {
          textEl.removeAttribute('data-dyn-key');
          // The placeholder was written via ``textContent`` (a raw text
          // node), but ``setBubbleText`` goes through the streaming-text
          // helper which only manages ``.text-frame`` siblings. Without
          // a wipe here the helper would *append* the final transcript
          // beside the leftover "识别中…" text and the user would see
          // them glued together. Clearing the element back to an empty
          // state gives streaming-text a clean canvas.
          textEl.textContent = '';
        }
      });
  }

  function updatePartialBubble(segId, text, textSecondary) {
    const wrapper = partialBubbles.get(segId);
    if (!wrapper) return false;
    // Upgrading from a "识别中…" placeholder to a real partial drops
    // the breathing indicator so the text doesn't keep pulsing while
    // it streams.
    clearRecognizingDecor(wrapper);
    setDualBubbleText(wrapper, text, textSecondary);
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    return true;
  }

  function discardPartialBubble(segId) {
    const wrapper = partialBubbles.get(segId);
    if (!wrapper) return;
    partialBubbles.delete(segId);
    if (wrapper.parentNode) wrapper.parentNode.removeChild(wrapper);
  }

  function addFinalBubble(text, textSecondary, langValue, durationSec, segId) {
    // If a partial / processing bubble for this id is already mounted,
    // upgrade it in place: drop the recognizing breath, replace text,
    // attach meta + replay button, and avoid triggering another fly-in.
    let wrapper = segId ? partialBubbles.get(segId) : null;
    if (wrapper) {
      partialBubbles.delete(segId);
      wrapper.removeAttribute('data-partial');
      clearRecognizingDecor(wrapper);
      setDualBubbleText(wrapper, text, textSecondary);
    } else {
      wrapper = buildBubbleSkeleton(segId, false);
      setDualBubbleText(wrapper, text, textSecondary);
      chatArea.appendChild(wrapper);
    }
    applyMeta(wrapper, langValue, durationSec);
    applyReplayButton(wrapper, segId);
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
  }

  function addErrorBubble(code, message) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-row chat-row-ai chat-bubble-float';
    const vars = { code: code || 'error', msg: message || '' };
    wrapper.innerHTML = `
      <div class="flex gap-3 max-w-2xl items-start">
        <div class="chat-avatar flex-shrink-0"
             style="background:var(--danger-soft); border-color:rgba(180,80,74,0.4); color:var(--danger)">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"
               style="color:var(--danger)">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
          </svg>
        </div>
        <div class="chat-bubble chat-bubble-ai text-sm" style="color:var(--danger)"
             data-dyn-key="tsasr.error.serverPrefix"
             data-dyn-vars='${escapeHtml(JSON.stringify(vars))}'>
          ${escapeHtml(t('tsasr.error.serverPrefix', vars))}
        </div>
      </div>
    `;
    chatArea.appendChild(wrapper);
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = String(text == null ? '' : text);
    return div.innerHTML;
  }

  // -------------------- Upload (transcription stage only) --------------------

  function setUploadStatus(state, key, vars) {
    if (!uploadStatus) return;
    if (!key) {
      uploadStatus.hidden = true;
      uploadStatus.textContent = '';
      uploadStatus.removeAttribute('data-state');
      currentUploadDyn = null;
      return;
    }
    uploadStatus.hidden = false;
    uploadStatus.dataset.state = state || 'info';
    currentUploadDyn = { key, vars: vars || null };
    uploadStatus.textContent = t(key, vars || undefined);
  }

  function setUploadBusy(busy) {
    isUploading = busy;
    if (uploadBtnLabel) {
      uploadBtnLabel.textContent = t(busy ? 'tsasr.upload.uploading' : 'tsasr.upload.label');
    }
    updateMicGate();
  }

  async function handleUploadFile(file) {
    if (!file) return;
    if (!enrollWavB64) {
      setUploadStatus('error', 'tsasr.upload.error.noEnroll');
      return;
    }
    if (isUploading || isRecordingLive || isEnrollRecording || isEnrollUploading) {
      setUploadStatus('error', 'tsasr.upload.error.busy');
      return;
    }
    const upload = window.AmphionAudioUpload;
    if (!upload) {
      setUploadStatus('error', 'tsasr.upload.error.unsupported');
      return;
    }

    setUploadBusy(true);
    setUploadStatus('info', 'tsasr.upload.decoding');

    let decoded;
    try {
      decoded = await upload.decodeFileToWavBytes(file, TARGET_SAMPLE_RATE);
    } catch (err) {
      console.error('Upload decode failed:', err);
      setUploadBusy(false);
      setUploadStatus('error', 'tsasr.upload.error.decode');
      return;
    }
    if (!decoded || !decoded.wav || !decoded.pcm.length) {
      setUploadBusy(false);
      setUploadStatus('error', 'tsasr.upload.error.empty');
      return;
    }

    let pcm = decoded.pcm;
    let wavBytes = decoded.wav;
    const totalSec = pcm.length / TARGET_SAMPLE_RATE;
    let trimmedNote = null;
    if (totalSec > TSASR_UPLOAD_MAX_SECONDS) {
      pcm = new Float32Array(
        pcm.subarray(0, Math.floor(TSASR_UPLOAD_MAX_SECONDS * TARGET_SAMPLE_RATE))
      );
      wavBytes = upload.encodeWavBytes(pcm, TARGET_SAMPLE_RATE);
      trimmedNote = totalSec.toFixed(1);
    }

    setUploadStatus('info', 'tsasr.upload.analyzing');
    uploadController = new AbortController();
    let result;
    try {
      result = await upload.postWavToEndpoint(
        '/api/tsasr/upload',
        wavBytes,
        {
          enrollment_wav_base64: enrollWavB64,
          // Comma-joined hotword list (matches the v3 SFT prompt format
          // used by ``backend.tsasr.prompt.build_tsasr_content``). The
          // server only forwards the list when ``tsasr_enable_hotwords``
          // is on; sending it always is harmless.
          hotwords: getEffectiveHotwords().join(','),
          // ``voice_traits`` is accepted by the server for backward
          // compatibility but never written into the prompt (see
          // ``backend/tsasr/prompt.py`` docstring).
          voice_traits: '',
        },
        { signal: uploadController.signal, fileName: file.name || 'upload.wav' }
      );
    } catch (err) {
      console.error('Upload request failed:', err);
      uploadController = null;
      setUploadBusy(false);
      const aborted = err && err.name === 'AbortError';
      const detail = err && err.message ? err.message : 'Upload failed';
      // Surface enrollment-validation errors with the same chat bubble the
      // WS path uses so users get a consistent failure mode regardless of
      // whether the bad enrollment came from the mic or a file.
      if (err && err.payload && typeof err.payload.detail === 'object') {
        const d = err.payload.detail;
        addErrorBubble(d.code || 'error', d.message || detail);
      } else if (!aborted) {
        addErrorBubble('upload_error', detail);
      }
      setUploadStatus(
        aborted ? 'info' : 'error',
        aborted ? 'tsasr.upload.aborted' : 'tsasr.upload.error.serverPrefix',
        aborted ? null : { msg: detail }
      );
      return;
    }
    uploadController = null;

    const text = ((result && result.text) || '').trim();
    const textSecondary = ((result && result.text_secondary) || '').trim();
    if (text || textSecondary) {
      // Mirror the WS ``final`` payload's replay-button wiring.
      if (result.audio_b64) {
        const synthId = `upload-${Date.now()}`;
        try {
          segmentAudio.set(synthId, b64ToWavBlobUrl(result.audio_b64));
        } catch (err) {
          console.warn('Failed to decode uploaded segment audio:', err);
        }
        const langValue =
          result.language && result.language !== 'N/A' ? result.language : null;
        const durationSec =
          typeof result.duration_sec === 'number' ? result.duration_sec : null;
        addFinalBubble(text, textSecondary, langValue, durationSec, synthId);
      } else {
        addFinalBubble(text, textSecondary, result.language || null, null, null);
      }
    }

    setUploadBusy(false);
    if (trimmedNote !== null) {
      setUploadStatus('warn', 'tsasr.upload.trimmed', {
        max: TSASR_UPLOAD_MAX_SECONDS,
        actual: trimmedNote,
      });
    } else {
      setUploadStatus('success', 'tsasr.upload.done');
    }
  }

  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener('click', () => {
      if (uploadBtn.disabled) return;
      uploadInput.value = '';
      uploadInput.click();
    });
    uploadInput.addEventListener('change', () => {
      const file = uploadInput.files && uploadInput.files[0];
      if (file) handleUploadFile(file);
    });
  }

  // -------------------- WebSocket --------------------
  function openWsAndStart() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/transcribe-target-streaming`;
    setConnStatus('pending');
    setEnrollStatus('pending', 'tsasr.enroll.sending');
    ws = new WebSocket(url);

    ws.onopen = () => {
      setConnStatus('connected');
      const payload = {
        type: 'start',
        format: 'pcm_s16le',
        sample_rate_hz: TARGET_SAMPLE_RATE,
        channels: 1,
        enrollment_audio: enrollWavB64,
        enrollment_format: 'wav',
        // Initial hotword snapshot. Mid-session edits are pushed via
        // ``update_hotwords`` (see ``saveAndSyncHotwords`` ->
        // ``syncHotwords``). When the toggle is off ``getEffectiveHotwords``
        // returns ``[]`` so the server effectively sees no hotwords.
        hotwords: getEffectiveHotwords(),
      };
      ws.send(JSON.stringify(payload));
      // The WS just opened so a synced status is now accurate. Echo it
      // to the pill (it won't reflect the open state until the first
      // client-driven action otherwise).
      setHotwordSyncStatus('synced');
    };

    ws.onmessage = (evt) => {
      let data;
      try {
        data = JSON.parse(evt.data);
      } catch {
        return;
      }
      handleServerMessage(data);
    };

    ws.onerror = () => {
      setConnStatus('disconnected');
      setHotwordSyncStatus('offline');
      if (extractRequestId) {
        extractRequestId = null;
        setExtractBusy(false);
        setExtractStatus('error', 'tsasr.extract.connError');
      }
    };

    ws.onclose = () => {
      setConnStatus('disconnected');
      ws = null;
      setHotwordSyncStatus('offline');
      if (extractRequestId) {
        extractRequestId = null;
        setExtractBusy(false);
        setExtractStatus('error', 'tsasr.extract.connClosed');
      }
      // Server has finished its post-stop work and dropped the
      // connection. Re-enable the mic so the user can start the next
      // utterance.
      isAwaitingFinalize = false;
      if (isRecordingLive) {
        stopLiveStreaming({ sendStop: false, reason: 'ws_closed' });
      } else {
        updateMicGate();
      }
    };
  }

  function handleServerMessage(data) {
    switch (data.type) {
      case 'ready':
        // Server accepted the WS; waiting for our start ack.
        break;
      case 'enrollment_ok':
        setEnrollStatus(
          'ready',
          'tsasr.enroll.ready',
          { dur: (data.duration_sec || enrollDurationSec).toFixed(1) }
        );
        startLiveStreaming();
        break;
      case 'processing':
        // The backend has accepted a VAD segment and is about to invoke
        // AmphionTSASR. Paint a placeholder bubble carrying the segment
        // id so the eventual ``final`` (or empty-final discard) can
        // upgrade / remove it in place.
        if (data.id) addProcessingBubble(data.id);
        break;
      case 'final': {
        const ptext = (data.text || '').trim();
        const sectext = (data.text_secondary || '').trim();
        if (ptext || sectext) {
          // Cache the mixed-audio blob *before* rendering so addFinalBubble
          // can mount the replay button in the initial DOM pass. Always
          // overwrite on collision (backend should mint unique ids, but if
          // anything ever reuses a key we want the freshest audio, not the
          // stale one — a stale cache hit would make the replay button play
          // audio from a completely different transcript).
          if (data.id && data.audio_b64) {
            const prev = segmentAudio.get(data.id);
            if (prev) {
              try { URL.revokeObjectURL(prev); } catch { /* noop */ }
            }
            try {
              segmentAudio.set(data.id, b64ToWavBlobUrl(data.audio_b64));
            } catch (err) {
              console.warn('Failed to decode segment audio:', err);
            }
          }
          const langValue =
            data.language && data.language !== 'N/A' ? data.language : null;
          const durationSec =
            typeof data.duration_sec === 'number' ? data.duration_sec : null;
          addFinalBubble(
            ptext,
            sectext,
            langValue,
            durationSec,
            data.id || null,
          );
        } else if (data.id) {
          // Both rows empty + an id is the backend's "discard the
          // partial bubble" signal — fired when the optional silence
          // gate suppresses an utterance whose partial(s) we already
          // painted (or both ASR paths returned empty text).
          discardPartialBubble(data.id);
        }
        break;
      }
      case 'partial': {
        // Pseudo-streaming partial. Backend reuses the same id across all
        // partials in an utterance and on the eventual final.
        //
        // Both rows empty (with a valid id) is the backend's "drop what
        // we've painted for this utterance" signal — fired by the
        // optional silence gate, or when both ASR paths returned empty.
        // The next non-empty partial will arrive under a fresh id and
        // mint a new bubble, so we simply drop the existing one and
        // otherwise stay quiet.
        const pid = data.id;
        if (!pid) break;
        const ptext = (data.text || '').trim();
        const sectext = (data.text_secondary || '').trim();
        if (!ptext && !sectext) {
          discardPartialBubble(pid);
          break;
        }
        if (!updatePartialBubble(pid, ptext, sectext)) {
          addPartialBubble(pid, ptext, sectext);
        }
        break;
      }
      case 'error': {
        const code = data.code || 'error';
        // Pill stays short -- show the short code only; the full message
        // lives in the error chat bubble below.
        const shortCode = code.replace(/^enrollment_/, '');
        setEnrollStatus(
          'error',
          'tsasr.enroll.errorCode',
          { code: shortCode, defaultValue: shortCode }
        );
        addErrorBubble(code, data.message || '');
        if (code.startsWith('enrollment_')) {
          // Enrollment was rejected: tear down WS, let user re-record.
          stopLiveStreaming({ sendStop: false, reason: 'enrollment_rejected' });
          if (ws) {
            try { ws.close(); } catch { /* noop */ }
          }
        }
        break;
      }
      case 'extract_hotwords_result': {
        if (!extractRequestId || data.request_id !== extractRequestId) break;
        extractRequestId = null;
        setExtractBusy(false);
        const merged = mergeExtractedHotwords(data.hotwords || []);
        setExtractStatus('success', 'tsasr.extract.added', {
          added: merged,
          total: (data.hotwords || []).length,
        });
        break;
      }
      case 'extract_hotwords_error': {
        if (!extractRequestId || data.request_id !== extractRequestId) break;
        extractRequestId = null;
        setExtractBusy(false);
        if (data.message) {
          setExtractStatus('error', 'tsasr.extract.raw', { msg: data.message });
        } else {
          setExtractStatus('error', 'tsasr.extract.failed');
        }
        break;
      }
    }
  }

  // -------------------- Live streaming --------------------
  async function startLiveStreaming() {
    if (isRecordingLive) return;
    try {
      const { ctx, node, mediaStream } = await openSixteenKContext();
      liveCtx = ctx;
      liveNode = node;
      liveStream = mediaStream;
    } catch (err) {
      console.error(err);
      addErrorBubble('mic_denied', t('tsasr.enroll.micAlert'));
      setConnStatus('disconnected');
      return;
    }

    liveNode.port.onmessage = (evt) => {
      if (evt.data.type !== 'audio') return;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const pcm16 = floatToPcm16(evt.data.samples);
      ws.send(pcm16.buffer);
    };

    isRecordingLive = true;
    micBtn.classList.add('recording');
    micIcon.setAttribute('fill', 'currentColor');
    pulseRings.forEach((r) => r.classList.add('active'));
    updateMicGate();
  }

  async function stopLiveStreaming({ sendStop, reason } = { sendStop: true }) {
    if (!isRecordingLive && !liveCtx) return;

    if (liveNode) {
      liveNode.port.onmessage = null;
      liveNode.disconnect();
      liveNode = null;
    }
    if (liveCtx) {
      try { await liveCtx.close(); } catch { /* noop */ }
      liveCtx = null;
    }
    if (liveStream) {
      liveStream.getTracks().forEach((tr) => tr.stop());
      liveStream = null;
    }

    isRecordingLive = false;
    micBtn.classList.remove('recording');
    micIcon.setAttribute('fill', 'none');
    pulseRings.forEach((r) => r.classList.remove('active'));

    if (sendStop && ws && ws.readyState === WebSocket.OPEN) {
      // The mic-stop click means "I'm done with this take, give me the
      // result". Send stop and DON'T close the WS — the server still
      // has to flush the buffered audio, run inference, and emit the
      // final. It will close the socket from its end after on_stop
      // returns. Closing here would truncate the in-flight inference
      // response and orphan any "识别中…" placeholder already on
      // screen. While we wait, ``isAwaitingFinalize`` keeps the mic
      // button disabled so the user can't double-fire stop.
      try { ws.send(JSON.stringify({ type: 'stop' })); } catch { /* noop */ }
      isAwaitingFinalize = true;
    } else if (ws && ws.readyState === WebSocket.OPEN && reason !== 'keep_ws') {
      // Cancel-style paths (ws_closed echo, enrollment rejected, etc):
      // no stop will be processed server-side, so close right away.
      try { ws.close(1000); } catch { /* noop */ }
    }

    updateMicGate();
  }

  micBtn.addEventListener('click', async () => {
    if (micBtn.disabled) return;
    if (isRecordingLive || (ws && ws.readyState === WebSocket.OPEN)) {
      await stopLiveStreaming({ sendStop: true });
    } else {
      if (!enrollWavB64) return;
      openWsAndStart();
    }
  });

  // -------------------- Refresh on language change --------------------
  function refreshDynamic() {
    if (enrollStatusDyn.key) {
      enrollStatusPill.textContent = t(
        enrollStatusDyn.key,
        enrollStatusDyn.vars || undefined
      );
    }
    enrollRecLabel.textContent = isEnrollRecording
      ? t('tsasr.enroll.stop')
      : t('tsasr.enroll.start');
    if (uploadBtnLabel) {
      uploadBtnLabel.textContent = t(isUploading ? 'tsasr.upload.uploading' : 'tsasr.upload.label');
    }
    if (currentUploadDyn && uploadStatus) {
      uploadStatus.textContent = t(currentUploadDyn.key, currentUploadDyn.vars || undefined);
    }
    if (enrollUploadBtnLabel) {
      enrollUploadBtnLabel.textContent = t(
        isEnrollUploading ? 'tsasr.enrollUpload.uploading' : 'tsasr.enrollUpload.label'
      );
    }
    if (currentEnrollUploadDyn && enrollUploadStatus) {
      enrollUploadStatus.textContent = t(
        currentEnrollUploadDyn.key,
        currentEnrollUploadDyn.vars || undefined,
      );
    }
    // Re-render hotword list (the trash button's aria-label uses an i18n
    // key) and the dynamic translation pills (sync status, extract
    // status, count). The pills already carry ``data-dyn-key`` attrs, so
    // the chatArea walk below would normally cover them, but they live
    // outside #chat-area — translate explicitly here.
    renderHotwords();
    setHotwordSyncStatus(currentSyncState);
    if (currentExtractDyn && currentExtractDyn.key) {
      setDynText(
        hotwordExtractStatus,
        currentExtractDyn.key,
        currentExtractDyn.vars || undefined,
      );
    }
    if (hotwordExtractBtn) {
      setDynText(
        hotwordExtractBtn,
        extractRequestId ? 'tsasr.hotword.extracting' : 'tsasr.hotword.extract',
      );
    }
    updateMicGate();
    // Walk dyn nodes inside the chat area to refresh transcript meta + errors.
    chatArea.querySelectorAll('[data-dyn-key]').forEach((el) => {
      const key = el.getAttribute('data-dyn-key');
      let vars = null;
      const raw = el.getAttribute('data-dyn-vars');
      if (raw) {
        try { vars = JSON.parse(raw); } catch { vars = null; }
      }
      if (key === 'tsasr.meta.lang' && vars && vars.lang) {
        el.textContent = t(key, { lang: langDisplayName(vars.lang) });
      } else {
        el.textContent = t(key, vars || undefined);
      }
    });
  }

  i18nUnsub = onLangChange(refreshDynamic);

  // -------------------- Init --------------------
  setEnrollStatus('idle', 'tsasr.enroll.notRecorded');
  updateMicGate();

    // -------------------- Dispose --------------------
    // Called by the SPA router before the tsasr <main> is swapped out.
    // Aborts the in-flight upload (if any), shuts down the live WS,
    // releases both AudioContext + microphone graphs (live + enroll),
    // cancels the enrollment progress timer, revokes every replay /
    // preview blob URL we've minted, and unsubscribes from i18n change
    // events. Mirrors the ``beforeunload`` cleanup.
    return function disposeTsasr() {
      try {
        window.removeEventListener('beforeunload', onBeforeUnload);
      } catch (_) { /* ignore */ }
      if (uploadController) {
        try { uploadController.abort(); } catch (_) { /* ignore */ }
        uploadController = null;
      }
      if (enrollTimerId !== null) {
        try { clearInterval(enrollTimerId); } catch (_) { /* ignore */ }
        enrollTimerId = null;
      }

      // Live (microphone -> WS) graph.
      if (liveNode) {
        try { liveNode.port.onmessage = null; } catch (_) { /* ignore */ }
        try { liveNode.disconnect(); } catch (_) { /* ignore */ }
        liveNode = null;
      }
      if (liveCtx) {
        try { liveCtx.close(); } catch (_) { /* ignore */ }
        liveCtx = null;
      }
      if (liveStream) {
        try {
          liveStream.getTracks().forEach((tr) => {
            try { tr.stop(); } catch (_) { /* ignore */ }
          });
        } catch (_) { /* ignore */ }
        liveStream = null;
      }
      isRecordingLive = false;

      // Enrollment recording graph (only running when the user pressed
      // Start enrollment and didn't release).
      if (enrollNode) {
        try { enrollNode.port.onmessage = null; } catch (_) { /* ignore */ }
        try { enrollNode.disconnect(); } catch (_) { /* ignore */ }
        enrollNode = null;
      }
      if (enrollCtx) {
        try { enrollCtx.close(); } catch (_) { /* ignore */ }
        enrollCtx = null;
      }
      if (enrollStream) {
        try {
          enrollStream.getTracks().forEach((tr) => {
            try { tr.stop(); } catch (_) { /* ignore */ }
          });
        } catch (_) { /* ignore */ }
        enrollStream = null;
      }
      isEnrollRecording = false;

      // The TS-ASR WebSocket survives across utterances, so close it
      // outside of stopLiveStreaming and only on dispose.
      if (ws) {
        try {
          ws.onopen = null;
          ws.onclose = null;
          ws.onerror = null;
          ws.onmessage = null;
          if (ws.readyState === WebSocket.OPEN
              || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }
        } catch (_) { /* ignore */ }
        ws = null;
      }

      // Reset hotword sync pill + abort any in-flight extraction so a
      // remount starts in a clean state. The hotword list itself stays
      // in localStorage and will be re-read on the next mount.
      try { setHotwordSyncStatus('waiting'); } catch (_) { /* ignore */ }
      if (extractRequestId) {
        extractRequestId = null;
        try { setExtractBusy(false); } catch (_) { /* ignore */ }
        try { setExtractStatus('idle', 'tsasr.extract.idle'); } catch (_) { /* ignore */ }
      }

      // Revoke replay blob cache (also clears active <audio> playback).
      try { clearSegmentAudio(); } catch (_) { /* ignore */ }

      // The enrollment preview audio holds another blob URL; let it go.
      if (enrollPreviewUrl) {
        try { URL.revokeObjectURL(enrollPreviewUrl); } catch (_) { /* ignore */ }
        enrollPreviewUrl = null;
      }
      try {
        if (enrollPreviewEl) {
          enrollPreviewEl.pause();
          enrollPreviewEl.removeAttribute('src');
        }
      } catch (_) { /* ignore */ }

      if (typeof i18nUnsub === 'function') {
        try { i18nUnsub(); } catch (_) { /* ignore */ }
        i18nUnsub = null;
      }
    };
  }

  window.AmphionPages = window.AmphionPages || {};
  window.AmphionPages.tsasr = { init: initTsasr };
})();
