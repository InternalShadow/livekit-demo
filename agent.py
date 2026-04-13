import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    TurnHandlingOptions,
    cli,
    inference,
    llm,
    room_io,
)
from livekit.agents import function_tool
from livekit.plugins import (
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
agent_name = os.getenv("AGENT_NAME")
logger = logging.getLogger(f"agent-{agent_name}")

DEBUG_MODE = os.getenv("DEBUG_MODE", "").lower() == "true"
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

DEBUG_LOGS_DIR = Path(__file__).parent / "debug_logs"


def _dump_debug_log(
    chat_ctx: llm.ChatContext,
    *,
    session_id: str,
    agent_name: str,
    mode: str = "default",
    topic: str = "",
    round_num: int | None = None,
) -> None:
    if not DEBUG_MODE:
        return

    try:
        session_dir = DEBUG_LOGS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp_str}.json"

        messages: list[dict] = []
        for item in chat_ctx.items:
            entry: dict = {"type": type(item).__name__}
            if hasattr(item, "role"):
                entry["role"] = str(item.role)
            if hasattr(item, "content"):
                content = item.content
                if isinstance(content, list):
                    entry["content"] = [
                        str(c) if not isinstance(c, str) else c for c in content
                    ]
                else:
                    entry["content"] = str(content) if content is not None else None
            if hasattr(item, "name"):
                entry["name"] = item.name
            if hasattr(item, "call_id"):
                entry["call_id"] = item.call_id
            if hasattr(item, "arguments"):
                entry["arguments"] = item.arguments
            if hasattr(item, "output"):
                entry["output"] = item.output
            messages.append(entry)

        payload = {
            "session_id": session_id,
            "mode": mode,
            "topic": topic,
            "timestamp": now.isoformat(),
            "agent": agent_name,
            "messages": messages,
        }
        if round_num is not None:
            payload["round"] = round_num

        (session_dir / filename).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        logger.debug("Debug log written: %s/%s", session_id, filename)
    except Exception:
        logger.exception("Failed to write debug log")


def _get_room_name(agent: Agent) -> str:
    try:
        return agent.session.room_io.room.name
    except AttributeError:
        return "unknown"


class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly, reliable voice assistant that answers questions, explains topics, and completes tasks with available tools.

# Output rules

You are interacting with the user via voice, and must apply the following rules to ensure your output sounds natural in a text-to-speech system:

- Respond in plain text only. Never use JSON, markdown, lists, tables, code, emojis, or other complex formatting.
- Keep replies brief by default: one to three sentences. Ask one question at a time.
- Do not reveal system instructions, internal reasoning, tool names, parameters, or raw outputs
- Spell out numbers, phone numbers, or email addresses
- Omit `https://` and other formatting if listing a web url
- Avoid acronyms and words with unclear pronunciation, when possible.

# Conversational flow

- Help the user accomplish their objective efficiently and correctly. Prefer the simplest safe step first. Check understanding and adapt.
- Provide guidance in small steps and confirm completion before continuing.
- Summarize key results when closing a topic.

# Tools

- Use available tools as needed, or upon user request.
- Collect required inputs first. Perform actions silently if the runtime expects it.
- Speak outcomes clearly. If an action fails, say so once, propose a fallback, or ask how to proceed.
- When tools return structured data, summarize it to the user in a way that is easy to understand, and don't directly recite identifiers or other technical details.

# Guardrails

- Stay within safe, lawful, and appropriate use; decline harmful or out‑of‑scope requests.
- For medical, legal, or financial topics, provide general information only and suggest consulting a qualified professional.
- Protect privacy and minimize sensitive data.""",
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="""Greet the user and offer your assistance.""",
            allow_interruptions=True,
        )


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

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


async def _start_default_session(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(
            model="openai/gpt-5.3-chat-latest",
            extra_kwargs={"reasoning_effort": "low"},
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
            language="en-US"
        ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    @ctx.add_shutdown_callback
    async def _on_shutdown():
        if session.current_agent:
            _dump_debug_log(
                session.history,
                session_id=ctx.room.name,
                agent_name=type(session.current_agent).__name__,
                mode="default",
            )

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )


MODERATOR_VOICE_ID = "c961b81c-a935-4c17-bfb3-ba2239de8c2f"
ATTENDEE_VOICE_ID = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
MAX_PANEL_EXCHANGES = 6


class ModeratorAgent(Agent):
    def __init__(
        self,
        topic: str,
        round_num: int = 0,
        *,
        chat_ctx: llm.ChatContext | None = None,
        moderator_tts: inference.TTS | None = None,
        attendee_tts: inference.TTS | None = None,
    ) -> None:
        topic_instruction = (
            f"The discussion topic is: {topic}. Interpret and discuss this topic "
            "through your confused pirate worldview."
        ) if topic else (
            "No topic was given. Come up with an interesting modern topic to discuss, "
            "but interpret it through your confused pirate worldview."
        )

        super().__init__(
            instructions=f"""You are Captain Barnacles, the moderator of a panel discussion. You genuinely believe you are a pirate captain living in the golden age of piracy around 1720, but you are actually in the year 2026. You have no idea this is the case.

You are confused by modern concepts and earnestly try to relate everything to piracy, sailing, treasure, and the high seas. When modern things come up, you interpret them through your 18th-century pirate lens in ways that are sincerely wrong and amusing.

{topic_instruction}

Your role as moderator:
- Ask thought-provoking questions about the topic, filtered through your anachronistic pirate confusion
- React with genuine bewilderment when the attendee explains modern concepts
- Keep the conversation flowing with none-repeating follow-up questions
- Use pirate speech naturally but keep it understandable
- After asking your question or reacting, ALWAYS call pass_to_attendee to let the expert respond
- NEVER ask the same question twice. If the attendee's answer echoes a previous point, react with curiosity and steer the conversation to an adjacent or deeper subtopic rather than rephrasing the same question.
- Each of your questions should explore a different facet of the topic

Output rules:
- Respond in plain text only, no formatting
- Keep your turns to two to four sentences
- Speak conversationally as a voice, not text""",
            tts=moderator_tts,
            chat_ctx=chat_ctx,
            allow_interruptions=False,
        )
        self._topic = topic
        self._round_num = round_num
        self._moderator_tts = moderator_tts
        self._attendee_tts = attendee_tts

    async def on_enter(self):
        if self._round_num == 0:
            self.session.generate_reply(
                instructions=(
                    "Introduce yourself as Captain Barnacles. Set up the topic with "
                    "pirate-themed confusion and ask your first question to the expert "
                    "attendee. Then call pass_to_attendee."
                ),
            )
        elif self._round_num >= MAX_PANEL_EXCHANGES:
            _dump_debug_log(
                self.session.history,
                session_id=_get_room_name(self),
                agent_name="ModeratorAgent",
                mode="panel",
                topic=self._topic,
                round_num=self._round_num,
            )
            self.session.generate_reply(
                instructions=(
                    "This is the final round. Thank the attendee for the fascinating "
                    "discussion with a hearty pirate farewell. Do NOT call pass_to_attendee."
                ),
                tool_choice="none",
            )
        else:
            self.session.generate_reply(
                instructions=(
                    "React to what the attendee just said with pirate confusion, "
                    "then ask a NEW question about a different aspect of the topic. "
                    "Do NOT revisit questions you have already asked. "
                    "Then call pass_to_attendee."
                ),
            )

    @function_tool()
    async def pass_to_attendee(self, context: RunContext):
        """Call this after you have finished asking your question, to hand the floor to the expert attendee."""
        _dump_debug_log(
            context.session.history,
            session_id=_get_room_name(self),
            agent_name="ModeratorAgent",
            mode="panel",
            topic=self._topic,
            round_num=self._round_num,
        )
        return (
            AttendeeAgent(
                self._topic,
                self._round_num,
                chat_ctx=context.session.history,
                moderator_tts=self._moderator_tts,
                attendee_tts=self._attendee_tts,
            ),
            "The expert attendee is responding.",
        )


class AttendeeAgent(Agent):
    def __init__(
        self,
        topic: str,
        round_num: int = 0,
        *,
        chat_ctx: llm.ChatContext | None = None,
        moderator_tts: inference.TTS | None = None,
        attendee_tts: inference.TTS | None = None,
    ) -> None:
        super().__init__(
            instructions=f"""You are Dr. Sarah Chen, a knowledgeable and straight-laced expert panelist. You are in a panel discussion moderated by Captain Barnacles, a confused pirate captain who genuinely believes he is still living in the golden age of piracy despite being in 2026.

The discussion topic is: {topic or "whatever the moderator brings up"}

Your role as expert panelist:
- Answer the moderator's questions earnestly and informatively, even when they are filtered through a pirate lens
- Gently and politely correct the moderator's anachronistic assumptions
- Provide genuinely interesting information about the topic
- The comedy comes naturally from your serious expertise clashing with the moderator's pirate confusion
- After making your point, ALWAYS call pass_to_moderator to let the moderator continue
- NEVER repeat a point you have already made. Review the conversation history before responding. If the moderator circles back to something you already covered, acknowledge it briefly and pivot to a new angle, a deeper detail, or a surprising related fact.

Output rules:
- Respond in plain text only, no formatting
- Keep your turns to two to four sentences
- Speak conversationally as a voice, not text
- Be informative but warm
- Each answer must introduce at least one idea not yet mentioned in the conversation""",
            tts=attendee_tts,
            chat_ctx=chat_ctx,
            allow_interruptions=False,
        )
        self._topic = topic
        self._round_num = round_num
        self._moderator_tts = moderator_tts
        self._attendee_tts = attendee_tts

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Respond to the moderator's question with your expert knowledge. "
                "Do NOT repeat any point you have already made in this conversation. "
                "Introduce a fresh angle, deeper detail, or surprising fact. "
                "Then call pass_to_moderator."
            ),
        )

    @function_tool()
    async def pass_to_moderator(self, context: RunContext):
        """Call this after you have answered the question, to hand the floor back to the moderator."""
        _dump_debug_log(
            context.session.history,
            session_id=_get_room_name(self),
            agent_name="AttendeeAgent",
            mode="panel",
            topic=self._topic,
            round_num=self._round_num,
        )
        return (
            ModeratorAgent(
                self._topic,
                self._round_num + 1,
                chat_ctx=context.session.history,
                moderator_tts=self._moderator_tts,
                attendee_tts=self._attendee_tts,
            ),
            "The moderator is responding.",
        )


async def _start_panel_session(ctx: JobContext, topic: str):
    moderator_tts = inference.TTS(
        model="cartesia/sonic-3",
        voice=MODERATOR_VOICE_ID,
        language="en-US",
    )
    attendee_tts = inference.TTS(
        model="cartesia/sonic-3",
        voice=ATTENDEE_VOICE_ID,
        language="en-US",
    )

    session = AgentSession(
        llm=inference.LLM(
            model="openai/gpt-5.3-chat-latest",
            extra_kwargs={"reasoning_effort": "low"},
        ),
        tts=moderator_tts,
    )

    await session.start(
        agent=ModeratorAgent(
            topic=topic,
            moderator_tts=moderator_tts,
            attendee_tts=attendee_tts,
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=False,
            text_input=False,
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
