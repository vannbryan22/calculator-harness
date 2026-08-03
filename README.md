# calculator-harness

A small agentic harness that drives a tool-using LLM (calculator, read/write
file, time) through a JSON-protocol loop with deterministic guards.

## Setup

To install dependencies:

```bash
bun install
```

### API key (required)

The harness talks to **any OpenAI-compatible provider**. Defaults to
**DeepSeek** (`deepseek-chat` @ `https://api.deepseek.com`); all three settings
are overridable via environment variables, so switching providers is a `.env`
edit, not a code change.

| Variable | Default | What it is |
|---|---|---|
| `LLM_API_KEY` | *(none — required)* | Your provider API key |
| `LLM_MODEL` | `deepseek-chat` | Model id |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible base URL |

Get a DeepSeek key from <https://platform.deepseek.com>, then create a `.env`
in the project root (Bun auto-loads it; `.env` is gitignored):

```ini
LLM_API_KEY=sk-your-deepseek-key-here
# optional overrides:
# LLM_MODEL=deepseek-chat
# LLM_BASE_URL=https://api.deepseek.com
```

Or set it in your shell instead of a file:

**PowerShell:** `$env:LLM_API_KEY="your-key-here"`
**Git Bash / Linux / macOS:** `export LLM_API_KEY="your-key-here"`

If the key is missing, the harness prints an error and exits before making any
request.

#### Pointing at another provider
```ini
# Zhipu bigmodel PaaS
LLM_API_KEY=your-zhipu-key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4.5-air
```
```ini
# Local Ollama (OpenAI-compatible shim)
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:3b
```

## To run

```bash
bun run src/index.ts
```

Type `exit` (or Ctrl+C / close the terminal) to quit. An empty line just
re-prompts.

This project was created using `bun init` in bun v1.3.14. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.
