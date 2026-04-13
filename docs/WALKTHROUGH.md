---
title: Building a Two-Mode Voice AI Agent on LiveKit (Deepgram Nova 3 + OpenAI + Cartesia Sonic 3)
published: false
tags: [voice-ai, livekit, deepgram, python, nextjs]
cover_image:
---

# Building a Two-Mode Voice AI Agent on LiveKit (Deepgram Nova 3 + OpenAI + Cartesia Sonic 3)

I wanted to understand the modern voice AI stack end-to-end — not just read about it. So I built a small demo that runs two very different interaction modes on a single LiveKit Agents worker: a normal "talk to an assistant" call, and an agent-vs-agent panel debate where you sit and listen to two personas argue via tool-based handoffs. Same worker, same room, branch decided by a `mode` field in the job metadata.

The bones came from LiveKit's official starter templates — `livekit-browser-create-agent` for the Python agent and `examples/agent-start-react` for the Next.js frontend. In 2026, saying "I used `agent-start-react`" is like saying "I used `create-next-app`" — it's the standard on-ramp, and I'm not going to pretend I wrote the SDK. What I did build on top of the scaffolds is what this post is about: the two-mode architecture, the JWT-metadata routing that lets one worker serve both, the panel-debate handoff logic with a shared `ChatContext`, and the STT → LLM → TTS pipeline configuration.

Repo: [github.com/InternalShadow/livekit-demo](https://github.com/InternalShadow/livekit-demo)

## Architecture at a glance

One Python worker. One entrypoint. Two behaviors, chosen at room-join time.

```
                            ┌──────────────────────────┐
  Browser (Next.js)          │     agent.py worker      │
  ┌──────────────┐           │                          │
  │ Welcome view │           │  entrypoint(ctx)         │
  │  - Call      │           │     │                    │
  │  - Panel     │           │     ▼                    │
  └──────┬───────┘           │  parse ctx.job.metadata  │
         │                   │     │                    │
         ▼                   │     ├─ mode=default ─┐   │
  POST /api/token            │     │                │   │
  { mode, topic }            │     └─ mode=moderated│   │
         │                   │                      │   │
         ▼                   │   ┌──────────────┐   │   │
  LiveKit token with         │   │ DefaultAgent │◄──┘   │
  RoomConfiguration          │   │ STT→LLM→TTS  │       │
  { agents: [{ metadata }]}  │   └──────────────┘       │
         │                   │                          │
         ▼                   │   ┌───────────────────┐  │
  LiveKit Cloud room ────────┼──▶│ ModeratorAgent ⇄  │  │
                             │   │ AttendeeAgent     │  │
                             │   │ (tool handoffs,   │  │
                             │   │  shared ChatCtx)  │  │
                             │   └───────────────────┘  │
                             └──────────────────────────┘
```

The frontend mints a LiveKit token and stuffs `{ mode, topic }` into agent metadata via `RoomConfiguration`. The worker joins the room, parses the metadata off the job, and picks which `AgentSession` to start. No separate processes, no extra infra.

## The Call mode pipeline: STT → LLM → TTS

Call mode is the canonical voice assistant loop. User speaks, agent transcribes, agent thinks, agent speaks back. In LiveKit Agents you wire that up by configuring an `AgentSession` with the three pieces:

```python
session = AgentSession(
    stt=inference.STT(model="deepgram/nova-3", language="en"),
    llm=inference.LLM(
        model="openai/gpt-5.3-chat-latest",
        extra_kwargs={"reasoning_effort": "low"},
    ),
    tts=inference.TTS(
        model="cartesia/sonic-3",
        voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
        language="en-US",
    ),
    turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
    vad=ctx.proc.userdata["vad"],
    preemptive_generation=True,
)
```

One thing worth being explicit about: I'm using LiveKit's `inference.*` module, not the vendor SDKs directly. That's the `livekit.agents.inference` proxy layer that routes to Deepgram / OpenAI / Cartesia through LiveKit's infrastructure. It means one auth path and one set of provider credentials managed at the LiveKit Cloud level instead of three. The trade-off is you lose some of the knobs you'd get calling Deepgram's streaming API directly — but for a demo that's a fair trade.

**How I picked the three models.** I'll be honest about the methodology: I blanket-tested the available STT/LLM/TTS options through LiveKit's inference layer and picked whichever combination sounded human enough and responded fast enough without running up the bill. Not a rigorous benchmark — a qualitative sweep. I did not build a latency harness, I did not load-test, and I stopped swapping things out once the round-trip felt conversational and the voice didn't make me wince.

**Deepgram Nova 3 for STT** ended up being the pick because it's Deepgram's current streaming model and, subjectively, the tail of the transcript lands fast enough that the downstream turn detector has something to work with almost immediately after the user stops talking. It also pairs cleanly with the multilingual turn-detection plugin I'm using downstream.

**`gpt-5.3-chat-latest` for the LLM** with `reasoning_effort: "low"` — because voice conversations need *fast* replies more than they need *deep* ones. A voice agent that pauses three seconds to think feels broken in a way that a chat agent pausing three seconds does not. Low reasoning effort on a chat-tuned model is my opinionated default for the duplex loop; if you wanted a reasoning-heavy assistant you'd tune this differently and probably accept the latency hit.

**Cartesia Sonic 3 for TTS** because, side-by-side against the other TTS options I tried, Sonic 3 produced audio that sounded like it had intent — pauses in roughly the right places, stress on roughly the right words — and started returning audio fast enough that LiveKit's playback pipeline could stream it without a visible gap after the LLM finished. No benchmark table from me — this is qualitative and based on listening.

LiveKit Agents wires these three into a single `AgentSession` that runs the full duplex loop: room audio in → STT → LLM → TTS → room audio out, interruption-aware, with the transcripts surfaced on the data channel so the frontend can show them live.

## VAD, turn detection, and preemptive generation

The three less-glamorous settings in that `AgentSession` block are actually where the conversational feel lives.

**Silero VAD.** Voice activity detection — is the user currently making speech sounds, or not. Silero is the standard choice in the LiveKit Agents ecosystem; I'm loading it once in `prewarm` so every job reuses the same model in memory:

```python
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm
```

Loading it in `prewarm` matters because VAD has to run on every audio frame — you don't want to pay cold-start on each session.

**Multilingual turn detection.** VAD tells you when sound stops. It does *not* tell you whether the user is *done talking*. Those are different questions: a half-second pause between clauses shouldn't hand the turn over, but a full sentence ending on "…what do you think?" should. The `MultilingualModel` turn detector from `livekit.plugins.turn_detector.multilingual` is a small model that classifies end-of-turn from the incoming transcript stream. It's what keeps the agent from cutting you off mid-sentence.

I didn't tune this one by hand. I'm running the plugin's defaults because they felt right in testing, and "felt right in testing" is an honest thing to say — I did not benchmark it against a handcrafted silence threshold or against alternative end-of-turn classifiers. If I were shipping this to a demanding production surface, that's where I'd invest next.

**Preemptive generation.** The `preemptive_generation=True` flag tells the session to start generating the LLM response before the turn detector is 100% sure the user is done. If the user keeps talking, the half-baked response is thrown away. If they're actually done, you've already saved a few hundred milliseconds of latency. It's a bet on the common case, and for this demo it's a bet I'm happy to make.

**BVC noise cancellation.** The `room_options` also wire up LiveKit's built-in noise cancellation, with a small branch that switches to the telephony variant for SIP participants:

```python
noise_cancellation=lambda params: (
    noise_cancellation.BVCTelephony()
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    else noise_cancellation.BVC()
),
```

No SIP in this demo, but it's a one-line change away and I left it in.

## The clever bit: routing two modes through JWT metadata

The most interesting architectural decision in this project is tiny. In the entrypoint:

```python
@server.rtc_session(agent_name=agent_name)
async def entrypoint(ctx: JobContext):
    metadata: dict = {}
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
        except (json.JSONDecodeError, TypeError):
            pass

    mode = metadata.get("mode", "default")

    if mode == "panel":
        await _start_panel_session(ctx, topic=metadata.get("topic", ""))
    else:
        await _start_default_session(ctx)
```

That's the whole router. One worker, two modes, one conditional at session start. The thing I like about this pattern: there is *no* second process, no second worker pool, no service discovery, no "which agent do I dispatch to" dance on the frontend. It's a single deployment with two behaviors.

The metadata originates on the frontend. The `/api/token` route mints the LiveKit token and bolts a `RoomConfiguration` onto it that carries the agent name and a JSON metadata blob:

```typescript
const agentMetadata =
  mode === 'panel'
    ? JSON.stringify({ mode: 'panel', topic })
    : JSON.stringify({ mode: 'default' });

const roomConfig = RoomConfiguration.fromJson(
  { agents: [{ agent_name: agentName, metadata: agentMetadata }] },
  { ignoreUnknownFields: true },
);
```

When the browser joins the room with that token, LiveKit reads the `RoomConfiguration`, dispatches to the named agent, and hands the metadata through as `ctx.job.metadata` on the Python side. End-to-end, "click Panel" → "worker runs panel mode" is just data moving down the JWT.

## Panel mode: tool handoffs and shared ChatContext

Panel mode is where it gets fun. Two agents — a `ModeratorAgent` and an `AttendeeAgent` — take turns in the same LiveKit room, the user is a silent observer, and the "hand off to the other agent" primitive is a function tool.

The personas:

- **Captain Barnacles** (moderator) — a confused 18th-century pirate captain who earnestly believes it's 1720, interprets every modern topic through a pirate lens, and gets sincerely bewildered when corrected.
- **Dr. Sarah Chen** (attendee) — a straight-laced expert who answers the moderator's questions accurately and gently corrects his anachronisms.

Each gets a distinct Cartesia voice (two different `voice` IDs on the same `cartesia/sonic-3` model), and each has its own system prompt shaping its role in the exchange.

The handoff mechanic is a function tool:

```python
@function_tool()
async def pass_to_attendee(self, context: RunContext):
    """Call this after you have finished asking your question, to hand the floor to the expert attendee."""
    _dump_debug_log(...)
    return (
        AttendeeAgent(
            self._topic,
            self._round_num,
            chat_ctx=context.session.history,  # <-- this line is the key
            moderator_tts=self._moderator_tts,
            attendee_tts=self._attendee_tts,
        ),
        "The expert attendee is responding.",
    )
```

Returning a new `Agent` instance from a tool tells the session to swap in a different agent as the active speaker. The attendee has an identical `pass_to_moderator` tool that hands control back, incrementing a round counter. When the moderator sees `round_num >= MAX_PANEL_EXCHANGES`, it delivers a closing remark and declines to pass (via `tool_choice="none"`), and the session ends naturally.

**The shared `ChatContext` is the load-bearing design decision.** Look at that `chat_ctx=context.session.history` line. Every time a handoff happens, the *entire running conversation* is threaded into the new agent's constructor. The alternative — letting each agent start with its own blank context — would mean Dr. Chen would have no memory of what Captain Barnacles just asked her, and vice versa. The debate would devolve into two monologues. Sharing the context makes them feel like they're actually in the same conversation, because from the model's perspective they *are* — it's one rolling message history with two sets of instructions pointed at it.

The silent-observer pattern is the other piece. In `_start_moderated_session` the room is configured with `audio_input=False, text_input=False`:

```python
await session.start(
    agent=ModeratorAgent(...),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=False,
        text_input=False,
    ),
)
```

The user joins the room, hears both agents through LiveKit's audio plane, but has no input channel. This is the right shape for "panel discussion you're watching" as opposed to "chat room you can interject in." The same two-agent handoff pattern would work with the user as a participant — I just chose not to for this demo, because the comedic rhythm of the two personas is better without a third voice cutting in.

## Gotchas and things I'd do differently

A few things tripped me up or that I'd change on a second pass:

- **`AGENT_NAME` must match across both `.env.local` files.** The root `.env.local` configures which agent name the Python worker registers under. The `demo-project/.env.local` configures which agent name the `/api/token` route embeds in the `RoomConfiguration`. If those drift, the frontend mints a token for an agent that never dispatches, the room sits empty, and the error surface is quiet. Obvious in hindsight. Not obvious when you're setting it up for the first time.
- **`python agent.py download-files` is easy to forget.** The VAD and turn-detection plugins need model weights fetched once at install time. If you skip that step you get a runtime error on first session rather than a setup-time failure, which is a small papercut but worth scripting around.
- **Debug logging is gitignored for a reason.** `DEBUG_MODE=true` writes every conversation into `debug_logs/<session_id>/*.json` so I can replay what the models actually said. That directory is in `.gitignore` and should stay there — voice conversations are PII-shaped and you do not want to accidentally commit them.
- **I have not load-tested this, and I'm not going to.** One worker, two modes, a handful of concurrent sessions during development. The reason I don't have numbers on where it falls over — or rigorous latency numbers for the STT/LLM/TTS combo — is that running the kind of sweeps that would produce real benchmarks would mean paying for the inference calls on my own dime. For a personal demo, the math doesn't work. If you fork this and you have a budget, the harness is obvious; I just don't have one here.

## Wrapping up

That's the whole shape: a single LiveKit Agents worker that branches on JWT metadata into either a Deepgram/OpenAI/Cartesia duplex call or a two-agent panel debate with tool-based handoffs and a shared `ChatContext`. The routing is three lines. The handoff is one `@function_tool` per agent. Most of the complexity is in the system prompts and in picking sensible defaults for VAD, turn detection, and preemptive generation.

Repo again: [github.com/InternalShadow/livekit-demo](https://github.com/InternalShadow/livekit-demo). Fork it, swap in your own personas, or tear out the panel mode entirely and bolt your own agents onto the routing pattern — that's what the scaffold is there for.

If you're working on voice AI and want to compare notes, I'm [@internalshadow](https://github.com/internalshadow) on GitHub.