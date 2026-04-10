# LiveKit Voice AI Agent Demo

A full-stack demo of real-time voice AI powered by [LiveKit Agents](https://docs.livekit.io/agents/), featuring two interaction modes:

- **Call mode** — Talk directly to a voice assistant (STT via Deepgram, LLM via OpenAI, TTS via Cartesia), streamed in real time.
- **Panel mode** — Watch two AI agents with distinct personas debate a topic using tool-based handoffs, while you observe as a silent audience member.

The Python backend (`agent.py`) implements both modes in a single worker. The Next.js frontend provides a polished UI with mode selection, audio visualizers, and a live chat transcript.

## Prerequisites

- **Python 3.12+**
- **Node.js 18.18+**
- **pnpm** (`npm install -g pnpm`)
- A **LiveKit Cloud** account (or self-hosted LiveKit server) — you'll need a URL, API key, and API secret from the [LiveKit dashboard](https://cloud.livekit.io/)

## Setup

### 1. Clone the repo

```bash
git clone "https://github.com/InternalShadow/livekit-demo.git"
cd livekit-demo
```

### 2. Configure environment variables

Copy the example env files and fill in your LiveKit credentials in both:

```bash
cp .env.example .env.local
cp demo-project/.env.example demo-project/.env.local
```

Both files need `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. The `AGENT_NAME` value must match across the two files so the frontend dispatches to the correct agent.

To enable debug logging (writes conversation history to `debug_logs/`), set `DEBUG_MODE=true` in the root `.env.local`.

### 3. Python agent setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python agent.py download-files
```

`download-files` pulls the model weights needed by the VAD and turn-detection plugins.

### 4. Frontend setup

```bash
cd demo-project
pnpm install
```

## Running

Start both processes in separate terminals:

**Terminal 1 — Agent**

```bash
python agent.py dev
```

**Terminal 2 — Frontend**

```bash
cd demo-project
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## How It Works

### Call mode (default)

1. The user selects "Talk to Agent" and clicks **Start call**.
2. The frontend calls `POST /api/token` with `mode: "call"`, minting a LiveKit access token whose `RoomConfiguration` dispatches to the Python agent.
3. The agent joins the room and runs a full-duplex conversational loop: STT (Deepgram Nova 3) → LLM (OpenAI) → TTS (Cartesia Sonic 3), with Silero VAD, multilingual turn detection, and preemptive generation.
4. Voice transcriptions and text chat messages appear in a live transcript.

### Panel mode

1. The user selects "Panel Discussion", optionally enters a topic, and clicks **Start panel**.
2. The token is minted with `mode: "panel"` and the topic in agent metadata. The room is configured with `audio_input=False` and `text_input=False` so the user is an observer only.
3. Two AI agents take turns via tool-based handoffs:
   - **Captain Barnacles** (ModeratorAgent) — a confused pirate who interprets modern topics through an 18th-century lens.
   - **Dr. Sarah Chen** (AttendeeAgent) — a knowledgeable expert who earnestly corrects the moderator's anachronisms.
4. Each agent has a distinct Cartesia voice. They share a single `ChatContext` across handoffs so the conversation stays coherent. The discussion runs for a configurable number of exchanges before the moderator delivers a closing remark.

### Routing

A single `agent.py` worker handles both modes. The entrypoint parses `mode` from the job metadata embedded in the JWT and branches to the appropriate session setup — no separate agent processes needed.

## Project Structure

```
├── agent.py              # Python agent: DefaultAgent, ModeratorAgent, AttendeeAgent
├── requirements.txt      # Python dependencies
├── .env.example          # Agent env template (AGENT_NAME, LiveKit creds, DEBUG_MODE)
├── debug_logs/           # JSON conversation logs (git-ignored, written when DEBUG_MODE=true)
└── demo-project/         # Next.js frontend
    ├── app/              # App Router pages & API routes (/api/token, /api/debug-logs)
    ├── app-config.ts     # Feature flags, branding, visualizer settings
    ├── components/       # UI components (shadcn/ui + Agents UI)
    │   ├── app/          # App shell: app.tsx, view-controller, welcome-view
    │   └── agents-ui/    # LiveKit Agents UI components (session, transcript, controls)
    ├── hooks/            # React hooks (audio visualizers, debug mode, error handling)
    ├── lib/              # Utilities (token source, config loading)
    └── .env.example      # Frontend env template
```
