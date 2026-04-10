import { NextResponse } from 'next/server';
import { mkdir, writeFile } from 'fs/promises';
import { join } from 'path';

const DEBUG_MODE = process.env.DEBUG_MODE;

export async function POST(req: Request) {
  if (process.env.NODE_ENV !== 'development') {
    return new NextResponse('Not available in production', { status: 403 });
  }

  if (DEBUG_MODE !== 'true') {
    return new NextResponse(null, { status: 204 });
  }

  try {
    const body = await req.json();
    const { roomName, messages } = body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return new NextResponse('No messages provided', { status: 400 });
    }

    const logDir = join(process.cwd(), '..', 'debug_logs');
    const safeName = (roomName ?? 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
    const roomDir = join(logDir, safeName);
    await mkdir(roomDir, { recursive: true });

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `frontend_${timestamp}.json`;

    const payload = {
      roomName: roomName ?? 'unknown',
      timestamp: new Date().toISOString(),
      messageCount: messages.length,
      messages,
    };

    await writeFile(join(roomDir, filename), JSON.stringify(payload, null, 2), 'utf-8');

    return NextResponse.json({ ok: true, filename });
  } catch (error) {
    if (error instanceof Error) {
      console.error('debug-logs error:', error);
      return new NextResponse(error.message, { status: 500 });
    }
    return new NextResponse('Unknown error', { status: 500 });
  }
}
