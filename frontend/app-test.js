(() => {
  'use strict';

  // AST v3 test page module ("实时语音识别（测试用）").
  //
  // UI mirrors the realtime ASR page (frontend/app.js) but the transport is
  // the iFlytek Tuling AST v3 envelope protocol (docs/tuling-ast-v3-protocol.md)
  // against a hard-coded remote backend, instead of the browser-demo /ws/audio
  // protocol. The differences that drive this rewrite:
  //
  //   * Audio is 16 kHz mono s16le PCM, base64-encoded inside a JSON envelope
  //     (payload.audio.audio), not a raw 48 kHz binary frame.
  //   * Session lifecycle is driven by header.status (0 first / 1 middle /
  //     2 last), and ONE WebSocket connection == ONE session (the server's
  //     AstV3Protocol never resets _inbound_started), so each recording opens
  //     a fresh connection.
  //   * Results arrive as a lattice (payload.result.ws[].cw[].w) with
  //     msgtype "Progressive" (partial) / "sentence" (final), sharing one
  //     segId per segment.
  //
  // AST v3 is ASR-only, so the emotion toggle, LLM hotword extraction,
  // segment replay, dual-ASR debug, file upload and target-speaker enrollment
  // have no protocol representation here. To keep the UI visually identical to
  // the realtime page (per the "full UI" choice), those controls stay in the
  // DOM but are disabled (greyed out) at init.

  // Same-origin WebSocket proxy (backend "/astv3-test-proxy") that bridges to
  // the remote AST v3 backend. Deriving the URL from location keeps it
  // same-origin, which sidesteps the browser's mixed-content block: an HTTPS
  // page (playground.amphion.top) yields wss://, plain HTTP/localhost yields
  // ws://. The real upstream address lives only in the backend, never here.
  const AST_V3_URL = (() => {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${scheme}//${window.location.host}/astv3-test-proxy`;
  })();

  function initAsrTest() {
    // --- i18n ---
    const i18n = window.Amphion && window.Amphion.i18n;
    const t = (key, vars) => (i18n ? i18n.t(key, vars) : (vars && vars.defaultValue) || key);
    const onLangChange = (fn) => (i18n ? i18n.onChange(fn) : () => {});

    // --- Dispose state ---
    let isDisposed = false;
    let i18nUnsub = null;

    // --- Session / connection state ---
    let ws = null;
    let audioCtx = null;
    let workletNode = null;
    let mediaStream = null;
    let isRecording = false;
    let startFrameSent = false;     // gate audio frames until status=0 is sent
    let sessionSeq = 0;             // bumped per recording; namespaces bubble ids
    let currentSessionSeq = 0;      // the seq the open ws belongs to
    let traceId = '';
    let bizId = '';
    let closeTimer = null;          // fallback close after stop if no terminal
    const doneSegs = new Set();     // segment ids already finalized

    // --- Hotword state ---
    let hotwords = [];
    let hotwordEnabled = localStorage.getItem('hotword_enabled') !== '0';

    const SYNC_PILL_BASE = 'status-pill';
    const HOTWORD_BUCKETS = ['auto', 'chinese', 'english', 'indonesian', 'thai'];
    const HOTWORDS_PER_LANG_MIGRATED = 'hotwords_per_lang_migrated';
    const UI_TO_API_LANG = {
      auto: 'N/A',
      chinese: 'Chinese',
      english: 'English',
      indonesian: 'Indonesian',
      thai: 'Thai',
    };

    function migrateLegacyHotwords() {
      if (localStorage.getItem(HOTWORDS_PER_LANG_MIGRATED) === '1') return;
      const legacy = localStorage.getItem('hotwords');
      if (legacy) {
        try {
          const arr = JSON.parse(legacy);
          if (Array.isArray(arr)) {
            HOTWORD_BUCKETS.forEach((b) => {
              if (localStorage.getItem(`hotwords_${b}`) === null) {
                localStorage.setItem(`hotwords_${b}`, JSON.stringify(arr));
              }
            });
          }
        } catch {
          /* ignore */
        }
      }
      localStorage.setItem(HOTWORDS_PER_LANG_MIGRATED, '1');
    }

    function readHotwordBucket(langForUi) {
      const raw = localStorage.getItem(`hotwords_${langForUi}`);
      if (raw === null) return [];
      try {
        const arr = JSON.parse(raw);
        return Array.isArray(arr) ? arr : [];
      } catch {
        return [];
      }
    }

    function writeHotwordBucket(langForUi, words) {
      localStorage.setItem(`hotwords_${langForUi}`, JSON.stringify(words));
    }

    function apiLangFromUi(langForUi) {
      return UI_TO_API_LANG[langForUi] || 'N/A';
    }

    migrateLegacyHotwords();
    let srcLangUi = localStorage.getItem('asr_src_lang') || 'auto';
    if (!HOTWORD_BUCKETS.includes(srcLangUi)) srcLangUi = 'auto';

    // --- DOM refs ---
    const micBtn = document.getElementById('mic-btn');
    const micIcon = document.getElementById('mic-icon');
    const micStatus = document.getElementById('mic-status');
    const pulseRings = document.querySelectorAll('.pulse-ring');
    const chatArea = document.getElementById('chat-area');
    const hotwordInput = document.getElementById('hotword-input');
    const hotwordAddBtn = document.getElementById('hotword-add-btn');
    const hotwordList = document.getElementById('hotword-list');
    const hotwordClearBtn = document.getElementById('hotword-clear-btn');
    const hotwordEnabledInput = document.getElementById('hotword-enabled');
    const hotwordSyncStatus = document.getElementById('hotword-sync-status');
    const hotwordCount = document.getElementById('hotword-count');
    const hotwordTextarea = document.getElementById('hotword-textarea');
    const hotwordExtractBtn = document.getElementById('hotword-extract-btn');
    const hotwordExtractStatus = document.getElementById('hotword-extract-status');
    const asrLangSelect = document.getElementById('asr-lang-select');
    const emotionToggle = document.getElementById('emotion-toggle');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadInput = document.getElementById('upload-input');
    const enrollUploadBtn = document.getElementById('enroll-upload-btn');
    const enrollFileInput = document.getElementById('enroll-file-input');
    const enrollRecordBtn = document.getElementById('enroll-record-btn');
    const enrollPlayBtn = document.getElementById('enroll-play-btn');
    const enrollClearBtn = document.getElementById('enroll-clear-btn');

    // --- Dynamic translation helpers ---
    function setDynText(el, key, vars) {
      if (!el) return;
      el.setAttribute('data-dyn-key', key);
      if (vars) {
        el.setAttribute('data-dyn-vars', JSON.stringify(vars));
      } else {
        el.removeAttribute('data-dyn-vars');
      }
      el.textContent = t(key, vars || undefined);
    }

    function applyDyn(root) {
      const scope = root || document;
      scope.querySelectorAll('[data-dyn-key]').forEach((el) => {
        const key = el.getAttribute('data-dyn-key');
        let vars = null;
        const rawVars = el.getAttribute('data-dyn-vars');
        if (rawVars) {
          try { vars = JSON.parse(rawVars); } catch { vars = null; }
        }
        el.textContent = t(key, vars || undefined);
      });
    }

    // --- Hotword management ---
    function sanitizeHotwords(sourceWords) {
      const result = [];
      (Array.isArray(sourceWords) ? sourceWords : []).forEach((item) => {
        const value = String(item || '').trim();
        if (!value || result.includes(value)) return;
        result.push(value);
      });
      return result;
    }

    function renderHotwords() {
      hotwordList.innerHTML = '';
      hotwords.forEach((word, idx) => {
        const tag = document.createElement('span');
        tag.className = 'hotword-pill';
        tag.innerHTML =
          `<span>${escapeHtml(word)}</span>` +
          `<button data-idx="${idx}" aria-label="${escapeHtml(t('asr.hotword.removeAria'))}">&times;</button>`;
        tag.querySelector('button').addEventListener('click', () => removeHotword(idx));
        hotwordList.appendChild(tag);
      });
      setDynText(hotwordCount, 'asr.hotword.count', { n: hotwords.length });
    }

    function getEffectiveHotwords() {
      return hotwordEnabled ? hotwords : [];
    }

    // Hotwords are applied per-session on the AST v3 first frame, so there is
    // no live sync channel. The pill just reflects whether hotwords will be
    // sent on the next recording.
    function refreshHotwordStatus() {
      if (!hotwordSyncStatus) return;
      hotwordSyncStatus.className = SYNC_PILL_BASE;
      const key = hotwordEnabled ? 'asr.sync.active' : 'asr.sync.paused';
      setDynText(hotwordSyncStatus, key);
      hotwordSyncStatus.dataset.state = hotwordEnabled ? 'ready' : 'waiting';
    }

    function saveAndSyncHotwords() {
      hotwords = sanitizeHotwords(hotwords);
      writeHotwordBucket(srcLangUi, hotwords);
      localStorage.setItem('hotwords', JSON.stringify(hotwords));
      renderHotwords();
      refreshHotwordStatus();
    }

    function addHotword(text) {
      const words = text
        .split(/[,，\n]/)
        .map((w) => w.trim())
        .filter((w) => w && !hotwords.includes(w));
      if (words.length === 0) return;
      hotwords.push(...words);
      saveAndSyncHotwords();
    }

    function removeHotword(idx) {
      hotwords.splice(idx, 1);
      saveAndSyncHotwords();
    }

    function clearHotwords() {
      hotwords = [];
      saveAndSyncHotwords();
    }

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

    hotwordClearBtn.addEventListener('click', clearHotwords);

    hotwordEnabledInput.checked = hotwordEnabled;
    hotwordEnabledInput.addEventListener('change', () => {
      hotwordEnabled = hotwordEnabledInput.checked;
      localStorage.setItem('hotword_enabled', hotwordEnabled ? '1' : '0');
      refreshHotwordStatus();
    });

    if (asrLangSelect) {
      asrLangSelect.value = srcLangUi;
      asrLangSelect.addEventListener('change', () => {
        const next = asrLangSelect.value;
        if (!HOTWORD_BUCKETS.includes(next)) return;
        writeHotwordBucket(srcLangUi, sanitizeHotwords(hotwords));
        srcLangUi = next;
        localStorage.setItem('asr_src_lang', srcLangUi);
        hotwords = sanitizeHotwords(readHotwordBucket(srcLangUi));
        localStorage.setItem('hotwords', JSON.stringify(hotwords));
        renderHotwords();
        refreshHotwordStatus();
      });
    }

    hotwords = sanitizeHotwords(readHotwordBucket(srcLangUi));
    localStorage.setItem('hotwords', JSON.stringify(hotwords));
    renderHotwords();
    refreshHotwordStatus();

    // --- Disable AST v3-unsupported controls (kept in DOM, greyed out) ---
    function disableUnsupported() {
      const tip = t('asrtest.unsupported', { defaultValue: 'Not supported by AST v3' });
      const mark = (el) => {
        if (!el) return;
        el.disabled = true;
        el.classList.add('is-disabled-astv3');
        el.title = tip;
        el.setAttribute('aria-disabled', 'true');
      };
      mark(emotionToggle);
      mark(hotwordTextarea);
      mark(hotwordExtractBtn);
      mark(uploadBtn);
      mark(uploadInput);
      mark(enrollUploadBtn);
      mark(enrollFileInput);
      mark(enrollRecordBtn);
      mark(enrollPlayBtn);
      mark(enrollClearBtn);
      if (hotwordExtractStatus) {
        hotwordExtractStatus.textContent = tip;
        hotwordExtractStatus.classList.add('is-disabled-astv3');
      }
    }
    disableUnsupported();

    // --- Connection status (sidebar dot) ---
    function setConnState(state) {
      if (window.AmphionSidebar && window.AmphionSidebar.setConnectionState) {
        window.AmphionSidebar.setConnectionState(state);
      }
    }
    setConnState('idle');

    // --- AST v3 framing ---
    function genId(prefix) {
      return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    }

    function floatToPcmB64(float32) {
      const buf = new ArrayBuffer(float32.length * 2);
      const view = new DataView(buf);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        // little-endian s16le, regardless of host byte order
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
      const upload = window.AmphionAudioUpload;
      return upload ? upload.bytesToBase64(new Uint8Array(buf)) : '';
    }

    function sendStartFrame() {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const frame = {
        header: { traceId, bizId, status: 0 },
        // 低延迟调参：首帧 asr_config 覆写仅对本连接生效、不落盘，字段属与 /transcribe-streaming 共用的覆写白名单（见 docs/tuling-ast-v3-protocol.md 配置覆写）。
        parameter: {
          asr_config: {
            language: apiLangFromUi(srcLangUi),
            vad_start_frames: 10,
            pseudo_stream_first_partial_ms: 100,
          },
        },
        payload: { audio: { audio: '' } },
      };
      const hw = getEffectiveHotwords();
      if (hw.length) {
        frame.payload.text = { text: hw.join(',') };
      }
      ws.send(JSON.stringify(frame));
    }

    function sendAudioFrame(b64) {
      if (!ws || ws.readyState !== WebSocket.OPEN || !startFrameSent) return;
      ws.send(JSON.stringify({
        header: { traceId, bizId, status: 1 },
        payload: { audio: { audio: b64 } },
      }));
    }

    function sendStopFrame() {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({
        header: { traceId, bizId, status: 2 },
        payload: { audio: { audio: '' } },
      }));
    }

    function closeWs() {
      if (closeTimer) {
        clearTimeout(closeTimer);
        closeTimer = null;
      }
      if (ws) {
        try {
          ws.onopen = null;
          ws.onclose = null;
          ws.onerror = null;
          ws.onmessage = null;
          if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }
        } catch (_) { /* ignore */ }
        ws = null;
      }
      startFrameSent = false;
    }

    // --- AST v3 inbound handling ---
    function latticeText(result) {
      const wsArr = Array.isArray(result.ws) ? result.ws : [];
      let s = '';
      wsArr.forEach((w) => {
        (Array.isArray(w.cw) ? w.cw : []).forEach((c) => {
          s += (c && c.w) || '';
        });
      });
      return s;
    }

    function handleServerMessage(frame) {
      const header = frame.header || {};
      if (typeof header.code === 'number' && header.code !== 0) {
        // Error frame carries no payload. Surface it as a terminal error
        // bubble for this session so the row is not left hanging.
        const errId = `${currentSessionSeq}-err-${Date.now()}`;
        addAIBubble(errId);
        updateAIBubble(errId, header.message || 'error', 'error');
        return;
      }

      const result = frame.payload && frame.payload.result;
      if (!result) {
        // Terminal frame may also arrive without a usable result body.
        if (header.status === 2) finishSession();
        return;
      }

      if (header.status === 2) {
        // End-of-session marker (ls=true, no ws). Close out the session.
        finishSession();
        return;
      }

      const segId = `${currentSessionSeq}-${result.segId}`;
      const text = latticeText(result);

      if (result.msgtype === 'Progressive') {
        if (doneSegs.has(segId)) return;
        if (!document.getElementById(`ai-${segId}`)) addAIBubble(segId);
        updateAIBubble(segId, text, 'streaming');
      } else if (result.msgtype === 'sentence' && !result.ls) {
        if (!document.getElementById(`ai-${segId}`)) addAIBubble(segId);
        doneSegs.add(segId);
        updateAIBubble(segId, text, 'done');
      }
    }

    function finishSession() {
      closeWs();
      if (!isRecording) setConnState('idle');
    }

    // --- Chat bubbles ---
    function addAIBubble(segId) {
      const wrapper = document.createElement('div');
      wrapper.className = 'chat-row chat-row-ai chat-bubble-float';
      wrapper.id = `ai-${segId}`;
      wrapper.innerHTML = `
        <div class="flex gap-3 max-w-2xl items-start">
          <div class="chat-avatar flex-shrink-0">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
            </svg>
          </div>
          <div class="chat-bubble chat-bubble-ai ai-content">
            <div class="bubble-shimmer">
              <div class="shimmer-lines">
                <div class="shimmer-line w-48 h-3 mb-2"></div>
                <div class="shimmer-line w-36 h-3 mb-2"></div>
                <div class="shimmer-line w-24 h-3"></div>
              </div>
            </div>
            <div class="bubble-content" hidden>
              <div class="flex items-start gap-2">
                <p class="text-sm leading-relaxed flex-1 bubble-text"></p>
                <span class="bubble-replay-slot"></span>
              </div>
            </div>
          </div>
        </div>
      `;
      chatArea.appendChild(wrapper);
      scrollChatToBottom();
    }

    function setBubbleText(textEl, text) {
      if (!textEl) return;
      const next = text == null ? '' : String(text);
      if (window.AmphionStreamingText && window.AmphionStreamingText.apply) {
        window.AmphionStreamingText.apply(textEl, next);
      } else {
        textEl.textContent = next;
      }
    }

    function showShimmer(content, show) {
      if (!content) return;
      const shimmer = content.querySelector('.bubble-shimmer');
      const body = content.querySelector('.bubble-content');
      if (shimmer) shimmer.hidden = !show;
      if (body) body.hidden = show;
    }

    function applyHotwordHighlights(textEl, text, words) {
      if (!textEl || !words || !words.length) return 0;
      const ranges = collectHotwordRanges(text, words);
      if (!ranges.length) return 0;
      const current = textEl.querySelector(':scope > .text-frame.is-current');
      if (!current) return 0;
      const source = String(text || '');
      let html = '';
      let prev = 0;
      for (const r of ranges) {
        if (r.start > prev) html += escapeHtml(source.substring(prev, r.start));
        html += `<mark class="is-hotword">${escapeHtml(source.substring(r.start, r.end))}</mark>`;
        prev = r.end;
      }
      if (prev < source.length) html += escapeHtml(source.substring(prev));
      current.innerHTML = html;
      return ranges.length;
    }

    function updateAIBubble(segId, text, status, _modelHotwords = null, _debugInfo = null) {
      const bubble = document.getElementById(`ai-${segId}`);
      if (!bubble) return;
      const content = bubble.querySelector('.ai-content');
      if (!content) return;

      if (status === 'streaming') {
        showShimmer(content, false);
        const textEl = content.querySelector('.bubble-text');
        setBubbleText(textEl, text || '');
        scrollChatToBottom();
        return;
      } else if (status === 'processing') {
        const textEl = content.querySelector('.bubble-text');
        const hasText = textEl && textEl.querySelector('.text-frame');
        if (!hasText) showShimmer(content, true);
        scrollChatToBottom();
        return;
      } else if (status === 'done') {
        showShimmer(content, false);
        const textEl = content.querySelector('.bubble-text');
        const finalText = text || '';
        const wordsForHighlight = Array.from(new Set(getEffectiveHotwords()));

        textEl.removeAttribute('data-dyn-key');
        textEl.removeAttribute('data-dyn-vars');
        textEl.style.fontStyle = '';
        textEl.style.color = '';
        setBubbleText(textEl, finalText);
        applyHotwordHighlights(textEl, finalText, wordsForHighlight);
      } else if (status === 'error') {
        showShimmer(content, false);
        const body = content.querySelector('.bubble-content');
        if (body) {
          body.hidden = false;
          const msg = text || '';
          body.innerHTML = `<p class="text-sm" style="color:var(--danger)"
                                    data-dyn-key="asr.errorPrefix"
                                    data-dyn-vars='${escapeHtml(JSON.stringify({ msg }))}'>${escapeHtml(t('asr.errorPrefix', { msg }))}</p>`;
        }
      }

      scrollChatToBottom();
    }

    function scrollChatToBottom() {
      requestAnimationFrame(() => {
        chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
      });
    }

    // --- Audio capture (16 kHz for AST v3) ---
    async function startRecording() {
      if (isRecording) return;

      // getUserMedia needs a secure context (HTTPS or http://localhost). Over
      // plain HTTP on a remote host (e.g. http://<ip>:8080) the browser leaves
      // navigator.mediaDevices undefined, so report that precisely instead of
      // the misleading "permission denied" alert.
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert(t('asrtest.mic.insecure'));
        return;
      }

      // Force-close any lingering session from a previous recording so each
      // recording maps to exactly one AST v3 session (status 0 -> 2).
      closeWs();
      doneSegs.clear();

      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: { ideal: 16000 },
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
      } catch (err) {
        alert(t('asr.mic.alert.denied'));
        return;
      }

      sessionSeq += 1;
      currentSessionSeq = sessionSeq;
      traceId = genId('web');
      bizId = genId('biz');
      startFrameSent = false;

      try {
        ws = new WebSocket(AST_V3_URL);
      } catch (err) {
        // Defensive: a same-origin wss:// (via the backend proxy) should not
        // throw synchronously. If the URL is somehow rejected, release the mic
        // we just grabbed and surface it instead of failing silently.
        alert(t('asrtest.ws.blocked'));
        if (mediaStream) {
          mediaStream.getTracks().forEach((tr) => { try { tr.stop(); } catch (_) { /* ignore */ } });
          mediaStream = null;
        }
        return;
      }
      setConnState('pending');
      ws.onopen = () => {
        sendStartFrame();
        startFrameSent = true;
        setConnState('listening');
      };
      ws.onmessage = (evt) => {
        try {
          handleServerMessage(JSON.parse(evt.data));
        } catch {
          /* ignore non-JSON */
        }
      };
      ws.onerror = () => {
        setConnState('error');
      };
      ws.onclose = () => {
        startFrameSent = false;
        if (!isRecording && !isDisposed) setConnState('idle');
      };

      // 16 kHz capture: the AudioContext resamples the mic input, so the
      // worklet emits 16 kHz frames directly (no server-side resample needed).
      audioCtx = new AudioContext({ sampleRate: 16000 });
      try {
        await audioCtx.audioWorklet.addModule('audio-processor.js?v=' + Date.now());
      } catch (err) {
        alert(t('asr.mic.alert.denied'));
        stopRecording();
        return;
      }
      if (isDisposed) { stopRecording(); return; }

      const source = audioCtx.createMediaStreamSource(mediaStream);
      workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');
      workletNode.port.onmessage = (evt) => {
        if (evt.data.type !== 'audio') return;
        if (!ws || ws.readyState !== WebSocket.OPEN || !startFrameSent) return;
        sendAudioFrame(floatToPcmB64(evt.data.samples));
      };
      source.connect(workletNode);
      workletNode.connect(audioCtx.destination);

      isRecording = true;
      micBtn.classList.add('recording');
      micIcon.setAttribute('fill', 'currentColor');
      setDynText(micStatus, 'asr.mic.listening');
      pulseRings.forEach((r) => r.classList.add('active'));
    }

    function stopRecording() {
      if (!isRecording && !workletNode && !audioCtx) return;

      if (workletNode) {
        workletNode.port.onmessage = null;
        try { workletNode.disconnect(); } catch (_) { /* ignore */ }
        workletNode = null;
      }
      // Signal end-of-session; the server flushes the trailing utterance and
      // replies with sentence(s) + a status=2 terminal frame, after which we
      // close. A fallback timer closes the socket if the terminal never lands.
      sendStopFrame();
      if (audioCtx) {
        try { audioCtx.close(); } catch (_) { /* ignore */ }
        audioCtx = null;
      }
      if (mediaStream) {
        mediaStream.getTracks().forEach((tr) => {
          try { tr.stop(); } catch (_) { /* ignore */ }
        });
        mediaStream = null;
      }

      isRecording = false;
      micBtn.classList.remove('recording');
      micIcon.setAttribute('fill', 'none');
      setDynText(micStatus, 'asr.mic.start');
      pulseRings.forEach((r) => r.classList.remove('active'));

      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        if (closeTimer) clearTimeout(closeTimer);
        closeTimer = setTimeout(() => { closeWs(); setConnState('idle'); }, 3000);
      } else {
        setConnState('idle');
      }
    }

    micBtn.addEventListener('click', () => {
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    });

    // --- Utilities ---
    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    function escapeRegExp(text) {
      return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function collectHotwordRanges(text, candidateHotwords) {
      const source = String(text || '');
      const active = (Array.isArray(candidateHotwords) ? candidateHotwords : [])
        .map((w) => String(w || '').trim())
        .filter(Boolean);
      if (!source || active.length === 0) return [];

      const raw = [];
      active.forEach((word) => {
        const re = new RegExp(escapeRegExp(word), 'gi');
        let match = re.exec(source);
        while (match) {
          raw.push({ start: match.index, end: match.index + match[0].length });
          match = re.exec(source);
        }
      });
      if (!raw.length) return [];

      raw.sort((a, b) => (a.start !== b.start ? a.start - b.start : b.end - a.end));

      const merged = [];
      raw.forEach((r) => {
        const last = merged[merged.length - 1];
        if (!last || r.start >= last.end) {
          merged.push(r);
        } else if (r.end > last.end) {
          last.end = r.end;
        }
      });
      return merged;
    }

    // --- Language change refresh ---
    i18nUnsub = onLangChange(() => {
      refreshHotwordStatus();
      setDynText(micStatus, isRecording ? 'asr.mic.listening' : 'asr.mic.start');
      applyDyn(document);
    });

    // --- Dispose ---
    return function disposeAsrTest() {
      isDisposed = true;
      if (workletNode) {
        try { workletNode.port.onmessage = null; } catch (_) { /* ignore */ }
        try { workletNode.disconnect(); } catch (_) { /* ignore */ }
        workletNode = null;
      }
      if (audioCtx) {
        try { audioCtx.close(); } catch (_) { /* ignore */ }
        audioCtx = null;
      }
      if (mediaStream) {
        try {
          mediaStream.getTracks().forEach((tr) => {
            try { tr.stop(); } catch (_) { /* ignore */ }
          });
        } catch (_) { /* ignore */ }
        mediaStream = null;
      }
      isRecording = false;
      closeWs();
      doneSegs.clear();
      if (typeof i18nUnsub === 'function') {
        try { i18nUnsub(); } catch (_) { /* ignore */ }
        i18nUnsub = null;
      }
    };
  }

  window.AmphionPages = window.AmphionPages || {};
  window.AmphionPages['asr-test'] = { init: initAsrTest };
})();
