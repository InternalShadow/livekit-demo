# LiveKit Voice & Text AI Agent Demo

A full-stack demo of a real-time voice and text AI agent powered by [LiveKit Agents](https://docs.livekit.io/agents/). The Python backend handles speech-to-text (Deepgram), LLM reasoning (OpenAI), and text-to-speech (Cartesia). The Next.js frontend provides a polished UI for connecting to the agent via browser.

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

Open [http://localhost:3000](http://localhost:3000) in your browser. Clicking "Connect" generates a room token via the local API route, creates a LiveKit room, and automatically dispatches the agent into it.

## How It Works

1. The Next.js app serves a single-page UI with audio controls and a chat transcript.
2. When a user clicks Connect, the frontend calls `POST /api/token` to mint a LiveKit access token and room configuration.
3. The token includes an agent dispatch directive, so LiveKit automatically routes the session to the Python agent registered under `AGENT_NAME`.
4. The agent joins the room and begins a conversational loop: it listens via STT, reasons with the LLM, and responds via TTS — all streamed in real time.
5. Text chat messages and voice transcriptions are both displayed in the UI transcript.

## Project Structure

```
├── agent.py              # Python voice agent (LiveKit Agents SDK)
├── requirements.txt      # Python dependencies
├── .env.example          # Agent env template
└── demo-project/         # Next.js frontend
    ├── app/              # App Router pages & API routes
    ├── components/       # UI components (shadcn/ui + agents-ui)
    ├── hooks/            # React hooks (audio visualizers, etc.)
    ├── lib/              # Utilities
    └── .env.example      # Frontend env template
```
