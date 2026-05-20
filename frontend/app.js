(() => {
  'use strict';

  // ASR demo page module.
  //
  // Wrapped in an ``init`` factory so the SPA router (frontend/router.js)
  // can mount and tear down this page repeatedly within a single
  // document. ``init`` returns a ``dispose`` callback that the router
  // invokes before swapping the page out — that closes the WebSocket,
  // releases the AudioContext + microphone, revokes all the segment
  // blob URLs, aborts in-flight uploads, and unsubscribes from i18n
  // change events. None of this code runs on script load anymore;
  // everything is gated on the router calling ``init``.
  function initAsr() {
    // --- i18n ---
    const i18n = window.Amphion && window.Amphion.i18n;
    const t = (key, vars) => (i18n ? i18n.t(key, vars) : (vars && vars.defaultValue) || key);
    const onLangChange = (fn) => (i18n ? i18n.onChange(fn) : () => {});

    // --- Dispose state ---
    // ``connectWS``'s onclose schedules a reconnect setTimeout; if the
    // user navigates away mid-reconnect we'd otherwise leak the timer
    // and a fresh WebSocket. ``isDisposed`` short-circuits the
    // reconnect path; ``reconnectTimer`` is the handle we cancel from
    // dispose().
    let isDisposed = false;
    let reconnectTimer = null;
    let i18nUnsub = null;

    // --- State ---
    let ws = null;
  let audioCtx = null;
  let workletNode = null;
  let mediaStream = null;
  let isRecording = false;
  let hotwords = [];
  let hotwordEnabled = localStorage.getItem('hotword_enabled') !== '0';
  let emotionEnabled = localStorage.getItem('asr_emotion_enabled') === '1';
  let extractRequestId = null;
  let activeReplayAudio = null;
  const segmentAudio = new Map();
  const MAX_EXTRACTED_HOTWORD_LENGTH = 10;
  const partialSeqMap = new Map(); // utterance_id -> highest seq seen

  // Last-known UI states so we can re-render strings after a language switch.
  let currentSyncState = 'waiting';
  let currentExtractDyn = { key: 'asr.extract.idle', vars: null };

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

  function b64ToWavBlobUrl(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  }

  // --- DOM refs ---
  const SYNC_PILL_BASE = 'status-pill';

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
  const emotionToggleLabel = document.getElementById('emotion-toggle-label');
  const uploadBtn = document.getElementById('upload-btn');
  const uploadBtnLabel = uploadBtn ? uploadBtn.querySelector('.btn-upload-label') : null;
  const uploadInput = document.getElementById('upload-input');
  const uploadStatus = document.getElementById('upload-status');

  // Upload state. The upload path is a one-shot REST POST against
  // /api/asr/upload, so all we need to track is whether one is in flight
  // (to gate the mic button) plus the latest status text for re-rendering
  // on language switches.
  let isUploading = false;
  let uploadController = null;     // AbortController for in-flight fetch
  let currentUploadDyn = null;     // { key, vars } | null when hidden
  // Server caps ASR uploads at 60s (matches _ASR_MAX_SECONDS in main.py),
  // and the model itself was trained on 16 kHz mono — encoding to that
  // up-front saves the server a resample.
  const ASR_UPLOAD_SAMPLE_RATE = 16000;
  const ASR_UPLOAD_MAX_SECONDS = 60;

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

  function enforceHotwordLimit() {
    hotwords = sanitizeHotwords(hotwords);
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

  function setHotwordSyncStatus(state) {
    if (!hotwordSyncStatus) return;
    currentSyncState = state || 'waiting';
    hotwordSyncStatus.className = SYNC_PILL_BASE;
    if (state === 'synced') {
      const key = hotwordEnabled ? 'asr.sync.active' : 'asr.sync.paused';
      setDynText(hotwordSyncStatus, key);
      hotwordSyncStatus.dataset.state = hotwordEnabled ? 'ready' : 'waiting';
      return;
    }
    if (state === 'offline') {
      setDynText(hotwordSyncStatus, 'asr.sync.offline');
      hotwordSyncStatus.dataset.state = 'offline';
      return;
    }
    setDynText(hotwordSyncStatus, 'asr.sync.waiting');
    hotwordSyncStatus.dataset.state = 'waiting';
  }

  function syncHotwords() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'update_hotwords',
          hotwords: getEffectiveHotwords(),
          src_lang: apiLangFromUi(srcLangUi),
          enable_emotion: emotionEnabled,
        })
      );
      setHotwordSyncStatus('synced');
    } else {
      setHotwordSyncStatus('offline');
    }
  }

  function syncEmotionToggle() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'update_emotion',
          enabled: emotionEnabled,
        })
      );
    }
  }

  function refreshEmotionToggleLabel() {
    if (!emotionToggleLabel) return;
    const key = emotionEnabled ? 'asr.emotion.toggle.on' : 'asr.emotion.toggle.off';
    setDynText(emotionToggleLabel, key);
  }

  function saveAndSyncHotwords() {
    enforceHotwordLimit();
    writeHotwordBucket(srcLangUi, hotwords);
    localStorage.setItem('hotwords', JSON.stringify(hotwords));
    renderHotwords();
    syncHotwords();
  }

  function setExtractStatus(state, key, vars) {
    if (!hotwordExtractStatus) return;
    currentExtractDyn = { key, vars: vars || null };
    setDynText(hotwordExtractStatus, key, vars || undefined);
    hotwordExtractStatus.className = 'hotword-extract-status';
    if (state === 'loading') {
      hotwordExtractStatus.classList.add('is-loading');
    } else if (state === 'success') {
      hotwordExtractStatus.classList.add('is-success');
    } else if (state === 'error') {
      hotwordExtractStatus.classList.add('is-error');
    }
  }

  function setExtractBusy(busy) {
    if (!hotwordExtractBtn || !hotwordTextarea) return;
    hotwordExtractBtn.disabled = busy;
    setDynText(
      hotwordExtractBtn,
      busy ? 'asr.hotword.extracting' : 'asr.hotword.extract'
    );
    hotwordTextarea.disabled = busy;
    updateExtractButtonAttention();
  }

  function updateExtractButtonAttention() {
    if (!hotwordExtractBtn || !hotwordTextarea) return;
    const hasText = hotwordTextarea.value.trim().length > 0;
    hotwordExtractBtn.classList.toggle(
      'is-attention',
      hasText && !hotwordExtractBtn.disabled
    );
  }

  function mergeExtractedHotwords(words) {
    const normalized = Array.isArray(words)
      ? words
          .map((w) => String(w || '').trim())
          .filter((w) => w && w.length < MAX_EXTRACTED_HOTWORD_LENGTH)
      : [];
    if (normalized.length === 0) return { added: 0, total: 0 };
    let added = 0;
    normalized.forEach((word) => {
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
    return { added, total: normalized.length };
  }

  function requestHotwordExtraction(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setExtractStatus('error', 'asr.extract.wsOffline');
      return;
    }
    const payloadText = String(text || '').trim();
    if (!payloadText) {
      setExtractStatus('error', 'asr.extract.pasteFirst');
      return;
    }
    if (extractRequestId) {
      setExtractStatus('error', 'asr.extract.alreadyRunning');
      return;
    }

    extractRequestId = `extract-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    setExtractBusy(true);
    setExtractStatus('loading', 'asr.extract.loading');
    ws.send(
      JSON.stringify({
        type: 'extract_hotwords',
        request_id: extractRequestId,
        text: payloadText,
      })
    );
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
  hotwordExtractBtn.addEventListener('click', () => {
    requestHotwordExtraction(hotwordTextarea.value);
  });
  hotwordTextarea.addEventListener('input', updateExtractButtonAttention);

  hotwordEnabledInput.checked = hotwordEnabled;
  hotwordEnabledInput.addEventListener('change', () => {
    hotwordEnabled = hotwordEnabledInput.checked;
    localStorage.setItem('hotword_enabled', hotwordEnabled ? '1' : '0');
    syncHotwords();
  });

  if (emotionToggle) {
    emotionToggle.checked = emotionEnabled;
    refreshEmotionToggleLabel();
    emotionToggle.addEventListener('change', () => {
      emotionEnabled = emotionToggle.checked;
      localStorage.setItem('asr_emotion_enabled', emotionEnabled ? '1' : '0');
      refreshEmotionToggleLabel();
      syncEmotionToggle();
    });
  }

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
      syncHotwords();
    });
  }

  hotwords = sanitizeHotwords(readHotwordBucket(srcLangUi));
  localStorage.setItem('hotwords', JSON.stringify(hotwords));
  renderHotwords();
  setHotwordSyncStatus('waiting');
  setExtractStatus('idle', 'asr.extract.idle');
  updateExtractButtonAttention();

  // --- Connection status ---
  function setConnected(connected) {
    if (window.AmphionSidebar && window.AmphionSidebar.setConnectionState) {
      if (connected) {
        window.AmphionSidebar.setConnectionState('connected');
      } else {
        window.AmphionSidebar.setConnectionState('error', t('common.disconnected'));
      }
    }
  }

  // --- WebSocket ---
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws/audio`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setConnected(true);
      syncHotwords();
    };

    ws.onclose = () => {
      setConnected(false);
      setHotwordSyncStatus('offline');
      if (extractRequestId) {
        extractRequestId = null;
        setExtractBusy(false);
        setExtractStatus('error', 'asr.extract.connClosed');
      }
      // The upload path is REST and not bound to the WS lifecycle, so
      // there is nothing to clean up here for it.
      stopRecording();
      if (!isDisposed) {
        reconnectTimer = setTimeout(connectWS, 2000);
      }
    };

    ws.onerror = () => {
      setConnected(false);
      setHotwordSyncStatus('offline');
      if (extractRequestId) {
        extractRequestId = null;
        setExtractBusy(false);
        setExtractStatus('error', 'asr.extract.connError');
      }
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        handleServerMessage(data);
      } catch {
        // ignore non-JSON
      }
    };
  }

  function handleServerMessage(data) {
    switch (data.type) {
      case 'partial_transcript': {
        // Pseudo-streaming partial. We only need a single AI bubble per
        // utterance — the right-side user bubble was removed so the
        // realtime page mirrors the target-speaker page (one column of
        // assistant bubbles, replay control lives inside).
        const uid = data.utterance_id;
        if (!uid) break;
        const prevSeq = partialSeqMap.get(uid) || 0;
        if (typeof data.seq === 'number' && data.seq <= prevSeq) break;
        partialSeqMap.set(uid, data.seq || 0);

        if (!document.getElementById(`ai-${uid}`)) {
          addAIBubble(uid);
        }
        updateAIBubble(uid, data.text, 'streaming');
        break;
      }
      case 'vad_event':
        if (data.event === 'segment_detected') {
          // Stash the segment audio for later replay — the actual replay
          // button is mounted in the bubble when the final response
          // arrives (see updateAIBubble 'done' branch). Mounting it
          // earlier would clutter the bubble while it's still visibly
          // streaming.
          if (data.audio_b64) {
            segmentAudio.set(data.id, b64ToWavBlobUrl(data.audio_b64));
          }
          if (!document.getElementById(`ai-${data.id}`)) {
            addAIBubble(data.id);
          }
        }
        break;
      case 'status':
        updateAIBubble(data.id, null, 'processing');
        break;
      case 'response':
        partialSeqMap.delete(data.id);
        updateAIBubble(data.id, data.text, 'done', data.model_hotwords, {
          textPrimary: data.text_primary,
          textSecondary: data.text_secondary,
          fusionMeta: data.fusion_meta,
          srcLangDetected: data.src_lang_detected,
          emotion: data.emotion,
        });
        break;
      case 'discard':
        partialSeqMap.delete(data.id);
        removeSegmentBubbles(data.id);
        break;
      case 'error':
        partialSeqMap.delete(data.id);
        updateAIBubble(data.id, data.message || '', 'error');
        break;
      case 'extract_hotwords_result':
        if (!extractRequestId || data.request_id !== extractRequestId) {
          break;
        }
        extractRequestId = null;
        setExtractBusy(false);
        {
          const merged = mergeExtractedHotwords(data.hotwords || []);
          setExtractStatus('success', 'asr.extract.added', {
            added: merged.added,
            total: merged.total,
          });
        }
        break;
      case 'extract_hotwords_error':
        if (!extractRequestId || data.request_id !== extractRequestId) {
          break;
        }
        extractRequestId = null;
        setExtractBusy(false);
        if (data.message) {
          // Backend-supplied free-form text wins over the generic label so
          // operators see the actual reason; we don't translate it.
          setExtractStatus('error', 'asr.extract.raw', { msg: data.message });
        } else {
          setExtractStatus('error', 'asr.extract.failed');
        }
        break;
    }
  }

  // --- Chat bubbles ---
  function replaySegment(segId, btn) {
    if (activeReplayAudio) {
      activeReplayAudio.pause();
      const prevBtn = document.querySelector('.replay-btn.is-playing');
      if (prevBtn) prevBtn.classList.remove('is-playing');
      if (activeReplayAudio._segId === segId) {
        activeReplayAudio = null;
        return;
      }
      activeReplayAudio = null;
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

  // Single-sided bubble skeleton, intentionally identical in shape to
  // the TS-ASR page (frontend/tsasr-app.js#buildBubbleSkeleton):
  //
  //   .ai-content
  //     .bubble-shimmer    <- visible while we're waiting on the model
  //     .bubble-content    <- visible once partial/final text exists
  //       .bubble-text     <- streaming-text helper writes <span class="ch"> here
  //       .bubble-replay-slot
  //     .bubble-meta-slot
  //
  // The shimmer + content split lets us swap states without rebuilding
  // the whole bubble (no chat-bubble-float animation re-runs) while
  // keeping the upload path's "model is thinking" placeholder.
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
            <div class="bubble-meta-slot"></div>
          </div>
        </div>
      </div>
    `;

    chatArea.appendChild(wrapper);
    scrollChatToBottom();
  }

  function removeSegmentBubbles(segId) {
    const ai = document.getElementById(`ai-${segId}`);
    if (!ai || !ai.parentNode) {
      const url = segmentAudio.get(segId);
      if (url) URL.revokeObjectURL(url);
      segmentAudio.delete(segId);
      return;
    }
    ai.classList.add('chat-bubble-discard');
    ai.addEventListener(
      'animationend',
      () => {
        if (ai.parentNode) ai.parentNode.removeChild(ai);
        const url = segmentAudio.get(segId);
        if (url) URL.revokeObjectURL(url);
        segmentAudio.delete(segId);
      },
      { once: true },
    );
  }

  function fusionLabel(scope, value) {
    if (!value) return '-';
    const key = `fusion.${scope}.${value}`;
    return t(key, { defaultValue: value });
  }

  function langDisplayName(value) {
    if (!value) return '';
    const v = String(value).trim();
    if (!v) return '';
    return t(`lang.name.${v}`, { defaultValue: v });
  }

  function renderDualAsrDebug(debugInfo) {
    if (!debugInfo) return '';
    const primary = String(debugInfo.textPrimary || '').trim();
    const secondary = String(debugInfo.textSecondary || '').trim();
    const meta = debugInfo.fusionMeta || null;
    if (!primary && !secondary) return '';

    const selected = meta && meta.selected ? escapeHtml(fusionLabel('selected', meta.selected)) : '-';
    const reason = meta && meta.reason ? escapeHtml(fusionLabel('reason', meta.reason)) : '-';
    const similarity =
      meta && typeof meta.similarity === 'number' ? String(meta.similarity) : '-';

    return `
      <div class="mt-3 rounded-lg border p-2 text-xs space-y-1"
           style="border-color:var(--line); background:var(--paper-sunk); color:var(--ink-mute)">
        <div class="text-[11px] text-faint" data-dyn-key="asr.debug.title">${escapeHtml(t('asr.debug.title'))}</div>
        <div><span class="text-faint" data-dyn-key="asr.debug.primary">${escapeHtml(t('asr.debug.primary'))}</span> ${escapeHtml(primary)}</div>
        <div><span class="text-faint" data-dyn-key="asr.debug.secondary">${escapeHtml(t('asr.debug.secondary'))}</span> ${escapeHtml(secondary)}</div>
        <div>
          <span class="text-faint" data-dyn-key="asr.debug.selected">${escapeHtml(t('asr.debug.selected'))}</span>
          <span data-dyn-key="fusion.selected.${escapeHtml(meta && meta.selected ? meta.selected : '')}"
                data-dyn-vars='${escapeHtml(JSON.stringify({ defaultValue: (meta && meta.selected) || '-' }))}'>${selected}</span>
          | <span class="text-faint" data-dyn-key="asr.debug.reason">${escapeHtml(t('asr.debug.reason'))}</span>
          <span data-dyn-key="fusion.reason.${escapeHtml(meta && meta.reason ? meta.reason : '')}"
                data-dyn-vars='${escapeHtml(JSON.stringify({ defaultValue: (meta && meta.reason) || '-' }))}'>${reason}</span>
          | <span class="text-faint" data-dyn-key="asr.debug.sim">${escapeHtml(t('asr.debug.sim'))}</span> ${similarity}
        </div>
      </div>
    `;
  }

  function renderEmotionMeta(emotion) {
    if (!emotion) return '';
    const ser = String(emotion.ser_label || '').trim();
    const secText = String(emotion.sec_text || '').trim();
    if (!ser && !secText) return '';

    const parts = [];
    if (ser) {
      parts.push(
        `<span class="text-faint">${escapeHtml(t('asr.emotion.result.ser'))}:</span> ${escapeHtml(ser)}`
      );
    }
    if (secText) {
      parts.push(
        `<span class="text-faint">${escapeHtml(t('asr.emotion.result.sec'))}:</span> ${escapeHtml(secText)}`
      );
    }
    return `
      <div class="text-[11px] mt-2" style="color:var(--accent-deep)">
        ${parts.join(' &middot; ')}
      </div>
    `;
  }

  // Route every text mutation through the diff helper. Fallback to plain
  // textContent guards against the script tag failing to load.
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

  function applyMeta(content, metaHtml) {
    const slot = content.querySelector('.bubble-meta-slot');
    if (!slot) return;
    if (!metaHtml) {
      slot.outerHTML = '<div class="bubble-meta-slot"></div>';
      return;
    }
    slot.outerHTML = `<div class="bubble-meta-slot mt-1 space-y-1">${metaHtml}</div>`;
  }

  function applyReplayButton(content, segId) {
    const slot = content.querySelector('.bubble-replay-slot');
    if (!slot) return;
    if (!segId || !segmentAudio.has(segId)) {
      slot.outerHTML = '<span class="bubble-replay-slot"></span>';
      return;
    }
    const replayTitle = escapeHtml(t('asr.user.replayTitle'));
    const btnHtml = `<button class="replay-btn bubble-replay-slot" type="button" title="${replayTitle}">
        <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
          <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
        </svg>
      </button>`;
    slot.outerHTML = btnHtml;
    const btn = content.querySelector('button.bubble-replay-slot');
    if (btn) {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        replaySegment(segId, e.currentTarget);
      });
    }
  }

  // Hotword highlighting for the crossfade representation.
  // ``setBubbleText`` swaps in a <span class="text-frame is-current"> with
  // plain text for every partial; on the final ``done`` event we reach
  // back into the now-stable current layer and rewrite its innerHTML so
  // matching substrings are wrapped in <mark class="is-hotword">. The
  // mark element naturally forms one continuous capsule per run, so no
  // boundary-detection passes are needed.
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

  function updateAIBubble(segId, text, status, modelHotwords = null, debugInfo = null) {
    const bubble = document.getElementById(`ai-${segId}`);
    if (!bubble) return;
    const content = bubble.querySelector('.ai-content');
    if (!content) return;

    if (status === 'streaming') {
      // Hide shimmer the moment the first partial arrives; from here on
      // the bubble grows char-by-char via the diff helper. We deliberately
      // do NOT highlight hotwords on partials — the wording shifts a lot
      // while the model decodes and a flickering highlight is jarring.
      showShimmer(content, false);
      const textEl = content.querySelector('.bubble-text');
      setBubbleText(textEl, text || '');
      scrollChatToBottom();
      return;
    } else if (status === 'processing') {
      // Server says "transcribing now". If we already have streaming
      // text, leave it in place — overwriting it with shimmer would
      // discard everything the user just watched type out. Only fall
      // back to shimmer when we have nothing to show (e.g. uploads,
      // or a segment that ended before any partial reached us).
      const textEl = content.querySelector('.bubble-text');
      const hasText = textEl && textEl.querySelector('.text-frame');
      if (!hasText) {
        showShimmer(content, true);
      }
      scrollChatToBottom();
      return;
    } else if (status === 'done') {
      showShimmer(content, false);
      const textEl = content.querySelector('.bubble-text');
      const finalText = text || '';
      setBubbleText(textEl, finalText);

      // Hotword feedback is now exclusively the inline <mark> highlight
      // applied by applyHotwordHighlights (which rewrites the current
      // text-frame's innerHTML once the final text has settled). The
      // previous "Hotword hits: N" line and the session-wide running
      // counter were redundant once the glyphs themselves are tinted,
      // and the bubble's meta block now matches the TS-ASR layout
      // (lang/duration only).
      const wordsForHighlight = Array.from(
        new Set([
          ...((Array.isArray(modelHotwords) ? modelHotwords : [])
            .map((w) => String(w || '').trim())
            .filter(Boolean)),
          ...getEffectiveHotwords(),
        ])
      );
      applyHotwordHighlights(textEl, finalText, wordsForHighlight);

      const detectedRaw =
        debugInfo && debugInfo.srcLangDetected
          ? String(debugInfo.srcLangDetected).trim()
          : '';
      const langDetectedMeta =
        detectedRaw && srcLangUi === 'auto'
          ? (() => {
              const vars = { lang: langDisplayName(detectedRaw) };
              return `<div class="text-[11px]" style="color:var(--info)"
                          data-dyn-key="asr.debug.langDetected"
                          data-dyn-vars='${escapeHtml(JSON.stringify({ lang: detectedRaw }))}'>${escapeHtml(t('asr.debug.langDetected', vars))}</div>`;
            })()
          : '';
      const debugBlock = renderDualAsrDebug(debugInfo);
      const emotionBlock = renderEmotionMeta(debugInfo && debugInfo.emotion);
      applyMeta(
        content,
        langDetectedMeta + emotionBlock + debugBlock,
      );
      applyReplayButton(content, segId);
    } else if (status === 'error') {
      // Wholesale replace the bubble body — the error is terminal for
      // this segment, no partial / replay context to preserve.
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

  // --- Audio capture ---
  async function startRecording() {
    if (isRecording) return;

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: { ideal: 48000 },
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (err) {
      alert(t('asr.mic.alert.denied'));
      return;
    }

    audioCtx = new AudioContext({ sampleRate: 48000 });
    await audioCtx.audioWorklet.addModule('audio-processor.js?v=' + Date.now());

    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');

    workletNode.port.onmessage = (evt) => {
      if (evt.data.type === 'audio' && ws && ws.readyState === WebSocket.OPEN) {
        const float32 = evt.data.samples;
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(int16.buffer);
      }
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
    if (!isRecording) return;

    // Detach the worklet's message port BEFORE we send the flush control
    // message so any audio frames the worklet had buffered can't sneak in
    // after our flush and end up split across two segments.
    if (workletNode) {
      workletNode.port.onmessage = null;
      workletNode.disconnect();
      workletNode = null;
    }
    // Tell the backend "no more audio coming, drain whatever you're
    // holding". Without this the trailing in-progress utterance sits in
    // VAD's buffer until the WebSocket actually closes, so the user
    // never sees a final bubble (and replay button) for the last thing
    // they said before clicking stop. Mirrors the TS-ASR ``stop``
    // protocol, see backend/session.py _handle_control_message.
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: 'flush' })); } catch { /* noop */ }
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }

    isRecording = false;
    micBtn.classList.remove('recording');
    micIcon.setAttribute('fill', 'none');
    setDynText(micStatus, 'asr.mic.start');
    pulseRings.forEach((r) => r.classList.remove('active'));
  }

  micBtn.addEventListener('click', () => {
    if (isUploading) return;
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  // --- Upload local audio file ---
  // The upload button hits POST /api/asr/upload; the response is a single
  // {text, language} payload that we render as one synthetic user/AI bubble
  // pair (ids namespaced as "upload-N" so they never collide with VAD ids
  // like "seg-3f9a-1").

  let uploadCounter = 0;

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
    if (uploadBtn) {
      uploadBtn.disabled = busy;
    }
    if (uploadBtnLabel) {
      setDynText(uploadBtnLabel, busy ? 'asr.upload.uploading' : 'asr.upload.label');
    }
    if (micBtn) {
      micBtn.disabled = busy || isRecording;
    }
  }

  async function handleUploadFile(file) {
    if (!file) return;
    if (isRecording) {
      alert(t('asr.upload.error.busyRecording'));
      return;
    }
    if (isUploading) return;

    const upload = window.AmphionAudioUpload;
    if (!upload) {
      setUploadStatus('error', 'asr.upload.error.unsupported');
      return;
    }

    setUploadBusy(true);
    setUploadStatus('info', 'asr.upload.decoding');

    let decoded;
    try {
      decoded = await upload.decodeFileToWavBytes(file, ASR_UPLOAD_SAMPLE_RATE);
    } catch (err) {
      console.error('Upload decode failed:', err);
      setUploadBusy(false);
      setUploadStatus('error', 'asr.upload.error.decode');
      return;
    }
    if (!decoded || !decoded.wav || !decoded.pcm.length) {
      setUploadBusy(false);
      setUploadStatus('error', 'asr.upload.error.empty');
      return;
    }

    let pcm = decoded.pcm;
    let wavBytes = decoded.wav;
    const totalSec = pcm.length / ASR_UPLOAD_SAMPLE_RATE;
    let trimmedNote = null;
    if (totalSec > ASR_UPLOAD_MAX_SECONDS) {
      // Match the server-side cap up-front so progress text matches what
      // the model actually transcribes; otherwise we'd display "60s sent"
      // for a 90s file and confuse the user when only the trailing window
      // came back transcribed.
      pcm = new Float32Array(
        pcm.subarray(0, Math.floor(ASR_UPLOAD_MAX_SECONDS * ASR_UPLOAD_SAMPLE_RATE))
      );
      wavBytes = upload.encodeWavBytes(pcm, ASR_UPLOAD_SAMPLE_RATE);
      trimmedNote = totalSec.toFixed(1);
    }

    // Stage a single AI bubble that immediately shows the shimmer while
    // the server is thinking — no companion user bubble (the realtime
    // page is now AI-only, replay button moves into the AI bubble once
    // the final text arrives).
    uploadCounter += 1;
    const segId = `upload-${uploadCounter}`;
    const audioB64 = upload.bytesToBase64(wavBytes);
    segmentAudio.set(segId, b64ToWavBlobUrl(audioB64));
    addAIBubble(segId);
    updateAIBubble(segId, null, 'processing');

    setUploadStatus('info', 'asr.upload.analyzing', {
      sec: (pcm.length / ASR_UPLOAD_SAMPLE_RATE).toFixed(1),
    });

    uploadController = new AbortController();
    const startedAt = performance.now();
    let result;
    try {
      result = await upload.postWavToEndpoint(
        '/api/asr/upload',
        wavBytes,
        {
          language: apiLangFromUi(srcLangUi) || '',
          hotwords: (hotwords || []).join(','),
        },
        { signal: uploadController.signal, fileName: file.name || 'upload.wav' }
      );
    } catch (err) {
      console.error('Upload request failed:', err);
      // Replace the "thinking" bubble with an error so the row is not left
      // hanging in a perpetually-spinning state.
      updateAIBubble(segId, err.message || 'Upload failed', 'error');
      setUploadBusy(false);
      uploadController = null;
      const key = err && err.name === 'AbortError'
        ? 'asr.upload.aborted'
        : 'asr.upload.error.request';
      setUploadStatus(err && err.name === 'AbortError' ? 'info' : 'error', key);
      return;
    }
    uploadController = null;

    const text = (result && result.text) || '';
    updateAIBubble(segId, text, 'done', undefined, {
      srcLangDetected: result && result.language,
    });

    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
    setUploadBusy(false);
    if (trimmedNote !== null) {
      setUploadStatus('warn', 'asr.upload.trimmed', {
        max: ASR_UPLOAD_MAX_SECONDS,
        actual: trimmedNote,
      });
    } else {
      setUploadStatus('success', 'asr.upload.done', { elapsed });
    }
  }

  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener('click', () => {
      if (isUploading) return;
      if (isRecording) {
        alert(t('asr.upload.error.busyRecording'));
        return;
      }
      uploadInput.value = '';
      uploadInput.click();
    });
    uploadInput.addEventListener('change', () => {
      const file = uploadInput.files && uploadInput.files[0];
      if (file) {
        handleUploadFile(file);
      }
    });
  }

  // --- Utilities ---
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // Returns merged, non-overlapping match ranges for ``text`` against
  // the supplied hotword list. Each range is in UTF-16 string offsets so
  // it can be compared against substring positions; ``applyHotwordHighlights``
  // then slices the text on those offsets and wraps each match in a
  // <mark class="is-hotword"> tag inside the current text-frame layer.
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

    // Sort by (start asc, length desc) so when two overlapping matches
    // share a start, the longer one wins the merge step below.
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
    setHotwordSyncStatus(currentSyncState);
    if (!isRecording) {
      setDynText(micStatus, 'asr.mic.start');
    } else {
      setDynText(micStatus, 'asr.mic.listening');
    }
    if (uploadBtnLabel) {
      setDynText(uploadBtnLabel, isUploading ? 'asr.upload.uploading' : 'asr.upload.label');
    }
    if (currentUploadDyn && uploadStatus) {
      uploadStatus.textContent = t(currentUploadDyn.key, currentUploadDyn.vars || undefined);
    }
    applyDyn(document);
  });

  // --- Init ---
  connectWS();

    // --- Dispose ---
    // Called by the SPA router before this page's <main> is replaced.
    // Must release every external resource the page has captured so we
    // don't leak across navigations:
    //   * WebSocket (and its scheduled reconnect timer)
    //   * AudioContext + AudioWorkletNode + the source MediaStream
    //   * The replay <audio> currently playing, if any
    //   * Every blob URL we minted via URL.createObjectURL for replay
    //   * Any in-flight upload fetch
    //   * The i18n change subscription
    return function disposeAsr() {
      isDisposed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
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
      if (activeReplayAudio) {
        try { activeReplayAudio.pause(); } catch (_) { /* ignore */ }
        activeReplayAudio = null;
      }
      segmentAudio.forEach((url) => {
        try { URL.revokeObjectURL(url); } catch (_) { /* ignore */ }
      });
      segmentAudio.clear();
      if (uploadController) {
        try { uploadController.abort(); } catch (_) { /* ignore */ }
        uploadController = null;
      }
      if (typeof i18nUnsub === 'function') {
        try { i18nUnsub(); } catch (_) { /* ignore */ }
        i18nUnsub = null;
      }
    };
  }

  window.AmphionPages = window.AmphionPages || {};
  window.AmphionPages.asr = { init: initAsr };
})();
