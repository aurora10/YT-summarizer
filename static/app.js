/* Front-end logic for the YouTube summarizer.
 *
 * Kept out of the template so the page can ship a strict CSP with no
 * 'unsafe-inline'. All model output goes through DOMPurify before it reaches
 * innerHTML: transcripts are third-party content, so treating the model's reply
 * as trusted HTML would be an XSS hole.
 */
(() => {
  'use strict';

  marked.setOptions({ gfm: true, breaks: true });

  const el = {
    history: document.getElementById('chat-history'),
    userMessage: document.getElementById('user_message'),
    send: document.getElementById('btn-send'),
    process: document.getElementById('btn-process'),
    clear: document.getElementById('btn-clear'),
    expand: document.getElementById('btn-expand'),
    videoUrl: document.getElementById('video_url'),
  };

  // Generous: a long transcript can take a while to summarize.
  const REQUEST_TIMEOUT_MS = 180_000;
  let busy = false;

  // ---------------------------------------------------------------- rendering

  function sanitizedHtml(text) {
    const raw = marked.parse(String(text ?? ''));
    // The app never needs model-authored inline styles, so dropping them removes
    // an entire class of CSS-injection tricks for free.
    return DOMPurify.sanitize(raw, {
      USE_PROFILES: { html: true },
      FORBID_ATTR: ['style'],
      FORBID_TAGS: ['style', 'form', 'input', 'button', 'textarea'],
    });
  }

  function appendMessage(role, text) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}-message`;

    const label = document.createElement('span');
    label.className = 'role';
    label.textContent =
      role === 'user' ? 'You:' : role === 'system' ? 'System:' : role === 'note' ? 'Note:' : 'LLM:';

    const body = document.createElement('div');
    body.className = 'message-body';

    if (role === 'llm' || role === 'system') {
      body.innerHTML = sanitizedHtml(text); // sanitized above
    } else {
      body.textContent = String(text ?? '');
    }

    wrapper.append(label, body);
    el.history.append(wrapper);
    scrollToBottom();
    return wrapper;
  }

  function appendNote(text) {
    return appendMessage('note', text);
  }

  function scrollToBottom() {
    el.history.scrollTop = el.history.scrollHeight;
  }

  function formatCount(n) {
    return typeof n === 'number' ? n.toLocaleString() : String(n);
  }

  // ------------------------------------------------------------------- state

  function setBusy(value) {
    busy = value;
    el.send.disabled = value;
    el.process.disabled = value;
    el.userMessage.disabled = value;
  }

  function setProcessing(label) {
    el.process.textContent = label || 'Summarize Video';
  }

  function describeError(err) {
    if (err && err.name === 'AbortError') {
      return 'The request timed out. Please try again with a shorter video.';
    }
    if (err && err.message) {
      return `Network error: ${err.message}`;
    }
    return 'Network error: could not reach the server.';
  }

  async function postJson(path, payload) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
        credentials: 'same-origin',
      });
      let data = null;
      try {
        data = await response.json();
      } catch {
        data = null;
      }
      return { response, data };
    } finally {
      clearTimeout(timer);
    }
  }

  // ----------------------------------------------------------------- actions

  async function processVideo() {
    if (busy) return;

    const url = el.videoUrl.value.trim();
    if (!url) {
      appendNote('Please enter a YouTube URL first.');
      el.videoUrl.focus();
      return;
    }

    setBusy(true);
    setProcessing('Processing…');

    let pending = appendMessage('system', '_Processing video transcript and generating summary…_');
    const clearPending = () => {
      if (pending) {
        pending.remove();
        pending = null;
      }
    };

    try {
      const { response, data } = await postJson('/api/process_video', { video_url: url });
      clearPending();

      if (response.ok && data && data.summary) {
        appendMessage('llm', data.summary);
        if (data.cache && data.cache.summary === 'hit') {
          appendNote(
            'Served from cache — this video was already summarized, so no transcript ' +
              'was fetched and no model call was made.'
          );
        } else if (data.cache && data.cache.transcript === 'hit') {
          appendNote(
            'Transcript loaded from cache; only the summary was generated just now.'
          );
        }
        if (data.truncated) {
          appendNote(
            `This summary covers only the first ${formatCount(data.transcript_chars)} ` +
              'characters of the transcript, because the video was too long to send in full.'
          );
        }
        el.userMessage.focus();
      } else {
        appendNote((data && data.error) || `Request failed (HTTP ${response.status}).`);
      }
    } catch (err) {
      clearPending();
      appendNote(describeError(err));
    } finally {
      setBusy(false);
      setProcessing('Summarize Video');
    }
  }

  async function sendMessage() {
    if (busy) return;

    const message = el.userMessage.value.trim();
    if (!message) return;

    el.userMessage.value = '';
    setBusy(true);

    appendMessage('user', message);

    let pending = appendMessage('system', '_LLM is thinking…_');
    const clearPending = () => {
      if (pending) {
        pending.remove();
        pending = null;
      }
    };

    try {
      const { response, data } = await postJson('/api/chat', { message });
      clearPending();

      if (response.ok && typeof (data && data.response) === 'string') {
        if (data.new_conversation) {
          appendNote(
            'Started a new conversation. Any earlier video context is no longer ' +
              'available — re-run "Summarize Video" for follow-up questions about it.'
          );
        }
        appendMessage('llm', data.response);
      } else {
        appendNote((data && data.error) || `Request failed (HTTP ${response.status}).`);
      }
    } catch (err) {
      clearPending();
      appendNote(describeError(err));
    } finally {
      setBusy(false);
      el.userMessage.focus();
    }
  }

  async function clearChat() {
    if (busy) return;
    el.history.replaceChildren();
    el.videoUrl.value = '';
    try {
      await fetch('/api/clear', { method: 'POST', credentials: 'same-origin' });
    } catch (err) {
      console.error('Failed to clear the session on the server', err);
    }
  }

  // ------------------------------------------------------------------ wiring

  el.send.addEventListener('click', sendMessage);
  el.process.addEventListener('click', processVideo);
  el.clear.addEventListener('click', clearChat);
  el.expand.addEventListener('click', () => {
    el.userMessage.value = 'Expand on: ';
    el.userMessage.focus();
  });

  el.userMessage.addEventListener('keydown', (event) => {
    // `busy` is checked in sendMessage too; this avoids the submit churn.
    if (event.key === 'Enter' && !event.shiftKey && !busy) {
      event.preventDefault();
      sendMessage();
    }
  });

  el.videoUrl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !busy) {
      event.preventDefault();
      processVideo();
    }
  });
})();
