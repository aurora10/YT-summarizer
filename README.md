# YouTube Chat & Summarizer

Paste a YouTube URL, get a bullet-point summary of the transcript, then ask
follow-up questions about the video in a chat.

The LLM backend is **provider-agnostic**: one env var picks the provider, one env
var holds the key. DeepSeek is the default.

**Contents:** [Requirements](#requirements) ·
[Run locally](#run-locally) · [Run with Docker](#run-with-docker) ·
[Verify the install](#verify-the-install) ·
[Switching LLM providers](#switching-llm-providers) ·
[Configuration](#configuration) · [Tests](#tests) · [Long videos](#long-videos) ·
[Caching](#caching) · [Architecture](#architecture) ·
[Troubleshooting](#troubleshooting)

---

## Requirements

| | Needed for | Notes |
|---|---|---|
| **Python 3.9+** | running locally | 3.12 recommended — that is what the Docker image uses. Check with `python3 --version`. |
| **An API key** | summarization | DeepSeek by default: create one at [platform.deepseek.com](https://platform.deepseek.com) → *API keys*. Any supported provider works; see below. |
| **Docker** *(optional)* | the container build | Only if you want to run it in a container instead of a venv. |

Nothing else is required: no database server, no Redis, no Node. The cache is a
local SQLite file that the app creates on first run.

There is no build step in the "compile the source" sense — Python is interpreted
and the browser assets are vendored. "Build" here means *create a virtualenv and
install dependencies*, or *build the Docker image*.

---

## Run locally

### 1. Get the code

```bash
cd ~/Desktop/CodeWork/YT-summarizer
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Two things that trip people up:

- Use **`python3`**, not `python`. On macOS only `python3` is on `PATH` by
  default. Once the venv is active, `python` points at the venv's interpreter.
- If this directory already has an `.venv` that came from somewhere else, delete
  it first (`rm -rf .venv`). A virtualenv hardcodes absolute paths, so one moved
  or copied from another directory has broken launchers — you get
  `cannot execute: required file not found` from `pip`.

Confirm you are in the venv — the prompt is prefixed with `(.venv)` and:

```bash
python -c "import sys; print(sys.prefix)"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
```

Open `.env` and set your key. The minimum for the default provider:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
```

`.env` must sit in the same folder as `app.py`; it is read at startup and is
already gitignored. To use a different provider, change `LLM_PROVIDER` — see
[Switching LLM providers](#switching-llm-providers).

Also worth setting now (optional locally, required in production):

```dotenv
SECRET_KEY=<generate one: python -c "import secrets; print(secrets.token_hex(32))">
```

### 5. Start the server

```bash
python app.py
```

You should see:

```
INFO  llm: LLM backend ready: DeepSeek (deepseek) model='deepseek-chat' key=set
INFO  __main__: Cache enabled at /.../cache.sqlite3
INFO  __main__: Listening on http://127.0.0.1:5001 (debug=False)
```

### 6. Use it

Open **http://127.0.0.1:5001**, paste a YouTube URL, press **Summarize Video**.

Start with a short video to confirm everything works, e.g.
`https://www.youtube.com/watch?v=jNQXAC9IVRw` (about 20 seconds). Then try a
long one.

You may also see a `NotOpenSSLWarning` from `urllib3` on Python 3.9 (macOS ships
LibreSSL). It is harmless; Python 3.11+ does not print it.

Press `Ctrl+C` to stop.

### Running it again later

```bash
cd ~/Desktop/CodeWork/YT-summarizer
source .venv/bin/activate
python app.py
```

### If macOS blocks outbound requests

If your machine routes traffic through a system HTTP proxy that cannot reach
YouTube, `yt-dlp` fails with `Unable to connect to proxy`. Bypass it for this app:

```bash
no_proxy='*' python app.py
```

Check what the system reports:

```bash
python -c "import urllib.request; print(urllib.request.getproxies())"
```

---

## Run with Docker

Build the image:

```bash
docker build -t yt-summarizer .
```

Run it — note the container listens on **8000**, not 5001:

```bash
docker run -d --name yt-summarizer \
  -p 8000:8000 \
  --env-file .env \
  -e SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
  -v yt-cache:/app/data \
  yt-summarizer
```

Open **http://127.0.0.1:8000**.

```bash
docker logs -f yt-summarizer    # follow logs: provider, cache state, timings
docker stop yt-summarizer       # stop
docker rm -f yt-summarizer      # remove
```

Each summarize logs a line such as
`Summarized dQw4w9WgXcQ in 3.2s (lang=en, 217 chars, cache=miss/miss)`, so you
can see hits, misses and timings without attaching a debugger.

The image runs gunicorn (not the Flask dev server) as a non-root user, and
reports health via `HEALTHCHECK` — `docker ps` shows `(healthy)` once it is up.

Two things that matter:

- **Mount the cache on a volume.** Without `-v`, `cache.sqlite3` lives inside the
  container filesystem and every redeploy starts cold — which also throws away
  the one thing that protects you from a YouTube block. The image already points
  `CACHE_DB` at `/app/data/cache.sqlite3` and creates that directory owned by the
  runtime user, so a named volume mounted there is writable out of the box.
- **One worker by design.** Conversations live in an in-process store, so
  multiple gunicorn *worker processes* would each see a different set of sessions.
  Concurrency comes from threads. Swap `sessions.SessionStore` for a Redis-backed
  implementation before raising `--workers`.

Port summary: **local `python app.py` → 5001**, **Docker → 8000**. Override the
local port with `PORT=8080 python app.py`.

---

## Verify the install

Any of these confirms the server is configured correctly. `curl` alone is enough
— you do not need to open a browser.

```bash
# 1. Is it up, and which backend did it pick?
curl -s localhost:5001/healthz | python -m json.tool
```

Expect `"status": "ok"`, your provider and model, `"key_configured": true`, and
cache counters. If the key is missing the app refuses to start and names the
variable to set.

```bash
# 2. Can it actually summarize? (fresh session cookie required)
curl -s -c /tmp/c.txt -o /dev/null localhost:5001/
curl -s -b /tmp/c.txt -H 'Content-Type: application/json' \
  -d '{"video_url":"https://www.youtube.com/watch?v=jNQXAC9IVRw"}' \
  localhost:5001/api/process_video | python -m json.tool
```

The answer includes `"cache": {"transcript": "miss", "summary": "miss"}`. Run the
same command again and both flip to `"hit"`, returning in milliseconds — that is
the cache working.

---

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

`GET /healthz` shows which provider and model are active.

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
| `MAX_TRANSCRIPT_CHARS` | `400000` | Absolute character ceiling for a transcript |
| `LLM_CONTEXT_TOKENS` | per provider | Context window; the transcript is clamped to fit |
| `SESSION_TTL_SECONDS` | `3600` | Conversation lifetime |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-session request budget |
| `CACHE_ENABLED` | `true` | Durable transcript + summary cache |
| `CACHE_DB` | `cache.sqlite3` | SQLite file for the cache |
| `TRANSCRIPT_CACHE_TTL` | `90 days` | Transcript reuse window |
| `FAILURE_CACHE_TTL` | `600` | How long a fetch failure is remembered |
| `PORT` | `5001` | Local listen port (`app.py` only) |
| `DEBUG` | `false` | Never enable on a reachable address |

## Tests

```bash
pip install pytest ruff     # dev tools, not runtime dependencies
python -m pytest
python -m ruff check .
```

The suite covers the pure pieces — URL parsing, subtitle parsers, transcript
truncation and token budgeting, the cache, the session store, the rate limiter,
and the LLM adapters against mocked HTTP responses — plus route-level tests with
the transcript fetcher and provider stubbed out. No network access or API key is
needed to run it.

## Long videos

A transcript is clamped to the smaller of two limits, because they catch
different problems:

- **`MAX_TRANSCRIPT_CHARS`** (400k) is an absolute ceiling — memory, storage, and
  absurd inputs.
- **The model's context window** is the constraint that actually matters. Token
  cost per character varies about 2× by script: 400k characters of English is
  roughly 100k tokens, while 400k of Russian is roughly 180k.

So the app estimates tokens for the transcript it actually has and shrinks the
character cap to fit `LLM_CONTEXT_TOKENS` (per provider, overridable). Truncation
is reported to the client and the model is told to mention it.

That means a long video never fails on context; the worst case is a summary of
the first part, clearly labelled. Measured on this project: a 130,874-character
Russian transcript (~55k tokens) summarizes whole on DeepSeek, which accepted
136k tokens in testing. Point the same app at an 8k-context local model and it
truncates to fit instead of erroring.

For content genuinely larger than the window, the upgrade is map-reduce
summarization: chunk the transcript, summarize each chunk, then summarize the
chunk summaries. Not implemented — it multiplies model calls by the chunk count.

## Caching

Two caches live in SQLite (`cache.py`), because the two expensive steps have very
different economics:

| Layer | Key | Why |
|---|---|---|
| Transcript | `video_id` | Effectively immutable. Reuse avoids the fetch that can be blocked. |
| Summary | `sha256(video_id, model, prompt_version, max_chars, lang)` | Tens of thousands of input tokens per call, identical for every user. |
| Failure | `video_id` | Short-lived, so a blocking provider is not hammered. |

What this buys you:

- **A YouTube block stops being fatal.** Videos already fetched keep working
  indefinitely; only new videos fail.
- **A repeat summarize makes zero network calls** and costs zero model tokens.
- **Raising `MAX_TRANSCRIPT_CHARS` invalidates nothing** — the full transcript is
  stored and truncated per request.
- **Follow-up questions still work on a cache hit.** The stored prompt and answer
  are replayed into a fresh conversation, so context is restored without a model
  call.

Change the prompt in `SUMMARY_SYSTEM_INSTRUCTION`? Bump `SUMMARY_PROMPT_VERSION`
in `app.py` and old summaries stop matching. Add `"force_refresh": true` to the
`/api/process_video` body to bypass both caches for a single request.

The cache is best-effort: if the database cannot be opened, the app logs it and
runs uncached rather than failing.

Delete `cache.sqlite3` at any time to start over; it is rebuilt automatically.

## Architecture

```
app.py          Flask app factory, routes, error → HTTP mapping, security headers
config.py       Environment → Settings; the provider registry lives here
llm.py          Provider-agnostic Chat interface + OpenAI/Gemini/Anthropic adapters
cache.py        SQLite caches for transcripts, summaries and recent failures
transcripts.py  Video id extraction, caption fetching, pure subtitle parsers
sessions.py     Bounded TTL session store + rate limiter
static/         app.js, style.css, vendor/ (pinned marked + DOMPurify)
templates/      index.html
tests/          pytest suite
Dockerfile      gunicorn image, non-root user, healthcheck
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
| `python: command not found` | macOS ships `python3` only. Use `python3 -m venv .venv`, then activate. |
| `pip: cannot execute: required file not found` | The `.venv` was created in a different directory (virtualenvs hardcode absolute paths). `rm -rf .venv` and recreate it. |
| `ModuleNotFoundError: No module named 'flask'` | The venv is not active, or you installed into a different interpreter. `source .venv/bin/activate`, then `pip install -r requirements.txt`. |
| `Configuration error: No API key found for provider 'deepseek'` | Set `DEEPSEEK_API_KEY` (or `LLM_API_KEY`) in `.env`, or set `LLM_PROVIDER=gemini` to use an existing `GOOGLE_API_KEY`. |
| `... rejected the request's credentials (401/400)` | The key does not match the selected provider. `GET /healthz` shows which provider and model are active. |
| `Could not reach YouTube from this server` | Network or proxy problem — not a bug in the app. See [If macOS blocks outbound requests](#if-macos-blocks-outbound-requests). |
| `yt-dlp could not read this video` / `Failed to extract any player response` | The `yt-dlp` release is too old for current YouTube: `pip install -U yt-dlp`. |
| Every video returns `no_transcript` | The video has no captions, or YouTube is blocking this IP. A `cookies.txt` (see `COOKIES_FILE`) helps with age/region-restricted videos. |
| Chat says "Started a new conversation" | The session expired (`SESSION_TTL_SECONDS`) or the server restarted. Re-run **Summarize Video**. |
| Summaries stop after the first N characters | The transcript was clamped to fit `MAX_TRANSCRIPT_CHARS` / `LLM_CONTEXT_TOKENS`; the summary says so. |
| `docker build` fails writing to `~/.docker` | Docker's buildx cache is not writable. Retry with the legacy builder: `DOCKER_BUILDKIT=0 docker build -t yt-summarizer .` |
| Container starts then restarts | Read `docker logs yt-summarizer` — usually a missing or wrong key, which fails startup on purpose. |
| Cache never fills up in Docker (`/app/data` stays empty) | The cache runs uncached rather than crashing, so this is silent. Check `docker exec yt-summarizer ls -l /app/data` — it must be owned by `appuser`. The image creates it correctly; a bind mount (`-v /host/path:/app/data`) can override that with the host's ownership. |

## Security notes

* Never commit `.env`. If a key has ever been committed, rotate it — removing the
  file from the latest commit does not remove it from history.
* `DEBUG=true` exposes the Werkzeug debugger, which is remote code execution.
  Keep it off and keep the port on localhost unless you know what you're doing.
