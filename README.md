# YouTube Chat & Summarizer

Paste a YouTube URL, get a bullet-point summary of the transcript, then ask
follow-up questions about the video in a chat.

The LLM backend is **provider-agnostic**: one env var picks the provider, one
env var holds the key.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# put your key in .env

python app.py            # http://127.0.0.1:5001
```

Open the page, paste a URL, press **Summarize Video**.

Requires Python 3.9+ (the Docker image uses 3.12). `requirements.txt` is complete
and installs cleanly into an empty virtualenv — that is verified by the test
suite, which also runs against a clean install.

> **Note on `yt-dlp`.** It is only the *fallback* transcript path, but it goes
> stale quickly because YouTube changes constantly. If summaries start failing
> with "The page needs to be reloaded" or "Failed to extract any player
> response", run `pip install -U yt-dlp`. The app reports the version it used and
> tells you when upgrading is the likely fix.

> **Upgrading from the previous version?** The app no longer reads
> `GOOGLE_API_KEY` implicitly. Either add `DEEPSEEK_API_KEY=...` (the new default
> provider), or keep using your existing Gemini key by setting
> `LLM_PROVIDER=gemini`. If the key is missing, startup fails with a message
> naming the variable to set.

## Switching LLM providers

Change `LLM_PROVIDER` and give it a key — nothing else needs to move.

```dotenv
# DeepSeek (default)
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-...

# ...or OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# ...or Google Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...

# ...or a local model, no key at all
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

Provider-specific keys (`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `OLLAMA_API_KEY`)
take precedence over the generic `LLM_API_KEY`. That means you can keep several
keys in `.env` and switch by editing `LLM_PROVIDER` alone.

| `LLM_PROVIDER` | Wire format | Default model |
|---|---|---|
| `deepseek` | OpenAI-compatible | `deepseek-chat` |
| `openai` | OpenAI-compatible | `gpt-4o-mini` |
| `gemini` | Google `generateContent` | `gemini-2.5-flash` |
| `anthropic` | Anthropic Messages | `claude-sonnet-4-5` |
| `openrouter` | OpenAI-compatible | `deepseek/deepseek-chat` |
| `groq` | OpenAI-compatible | `llama-3.3-70b-versatile` |
| `ollama` | OpenAI-compatible | `llama3.2` |
| `custom` | OpenAI-compatible | *(set `LLM_MODEL` + `LLM_BASE_URL`)* |

Verify what the server actually picked up:

```bash
curl -s localhost:5001/healthz | python -m json.tool
```

### Adding another provider

* **Speaks OpenAI, Gemini or Anthropic's format?** Add a `ProviderSpec` to
  `PROVIDERS` in `config.py`. No new code.
* **Something genuinely different?** Subclass `Provider` in `llm.py`, implement
  `_complete()`, and register it in `_KINDS`.

No provider SDKs are used — adapters call plain HTTP via `requests`, so timeouts,
retries and error mapping behave identically everywhere.

## Configuration

Everything is environment-driven; see [`.env.example`](.env.example) for the full
annotated list. The ones that matter most:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | Which backend to use |
| `LLM_API_KEY` | — | Generic key, fallback for any provider |
| `LLM_MODEL` | per provider | Override the model |
| `SECRET_KEY` | *ephemeral* | **Set this in production** — sessions and multi-worker safety depend on it |
| `MAX_TRANSCRIPT_CHARS` | `400000` | Longer transcripts are truncated, and the summary says so |
| `SESSION_TTL_SECONDS` | `3600` | Conversation lifetime |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-session request budget |
| `DEBUG` | `false` | Never enable on a reachable address |

## Running in production

```bash
docker build -t yt-summarizer .
docker run -p 8000:8000 --env-file .env -e SECRET_KEY=... yt-summarizer
```

The image runs gunicorn as a non-root user with a healthcheck.

> **One worker by design.** Conversations live in an in-process store, so multiple
> gunicorn *worker processes* would each see different sessions. Concurrency comes
> from threads. Swap `sessions.SessionStore` for a Redis-backed implementation
> before raising `--workers`.

## Tests

```bash
pip install pytest ruff
python -m pytest
python -m ruff check .
```

The suite covers the pure pieces — URL parsing, subtitle parsers, transcript
truncation, the session store, the rate limiter, and the LLM adapters against
mocked HTTP responses — plus route-level tests with the transcript fetcher and
provider stubbed out.

## Architecture

```
app.py          Flask app factory, routes, error → HTTP mapping, security headers
config.py       Environment → Settings; the provider registry lives here
llm.py          Provider-agnostic Chat interface + OpenAI/Gemini/Anthropic adapters
transcripts.py  Video id extraction, caption fetching, pure subtitle parsers
sessions.py     Bounded TTL session store + rate limiter
static/         app.js, style.css, vendor/ (pinned marked + DOMPurify)
```

Design notes worth knowing:

* **Model output is sanitized.** Transcripts are third-party content and the model
  echoes them, so replies are rendered through DOMPurify. The markdown libraries
  are vendored and pinned, which also allows a strict CSP with no `unsafe-inline`.
* **Transcripts are untrusted data.** The analyst role lives in a real system
  message; the transcript is fenced in the user turn and the model is told to
  treat any instructions inside it as content, not commands.
* **Failures are typed.** `TranscriptError` and `LLMError` subclasses map onto
  honest HTTP statuses (422 no transcript, 429 provider rate limit, 504 provider
  timeout, ...) instead of returning error text as a successful answer.
* **No TLS bypass.** Certificate verification is never disabled.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Configuration error: No API key found for provider 'deepseek'` | Set `DEEPSEEK_API_KEY` (or `LLM_API_KEY`), or set `LLM_PROVIDER=gemini` to use an existing `GOOGLE_API_KEY`. |
| `LLM_PROVIDER rejected the request's credentials (401/400)` | The key does not match the selected provider. `GET /healthz` shows which provider and model are active. |
| `Could not fetch the transcript: ... Unable to connect to proxy` | `yt-dlp` honours the system HTTP proxy. If the proxy cannot reach YouTube, bypass it with `no_proxy='*'` or configure the proxy properly. |
| `yt-dlp could not read this video` / `Failed to extract any player response` | The `yt-dlp` release is too old to parse current YouTube. Bump it (`pip install -U yt-dlp`) and re-run the tests. |
| Every video returns `no_transcript` | The video has no captions, or YouTube is blocking this IP. A `cookies.txt` (see `COOKIES_FILE`) helps with age/region-restricted videos. |
| Chat says "Started a new conversation" | The session expired (`SESSION_TTL_SECONDS`) or the server restarted. Re-run **Summarize Video**. |
| Summaries stop after the first N characters | `MAX_TRANSCRIPT_CHARS` truncated a long transcript; the summary tells you when that happened. |

## Security notes

* Never commit `.env`. If a key has ever been committed, rotate it — removing the
  file from the latest commit does not remove it from history.
* `DEBUG=true` exposes the Werkzeug debugger, which is remote code execution.
  Keep it off and keep the port on localhost unless you know what you're doing.
