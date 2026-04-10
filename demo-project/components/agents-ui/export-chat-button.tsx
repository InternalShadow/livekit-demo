'use client';

import { useCallback } from 'react';
import { Download } from 'lucide-react';
import type { ReceivedMessage } from '@livekit/components-react';
import { Button } from '@/components/ui/button';

function formatTranscript(messages: ReceivedMessage[]): string {
  const locale = typeof navigator !== 'undefined' ? navigator.language : 'en-US';

  const date = new Date().toLocaleDateString(locale, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const lines = messages.map((msg) => {
    const time = new Date(msg.timestamp).toLocaleTimeString(locale, {
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
    });
    const speaker = msg.from?.isLocal ? 'You' : 'Agent';
    return `[${time}] ${speaker}: ${msg.message}`;
  });

  return `LiveKit Demo - Chat Transcript\nDate: ${date}\n\n${lines.join('\n')}\n`;
}

export interface ExportChatButtonProps {
  messages: ReceivedMessage[];
  className?: string;
}

export function ExportChatButton({ messages, className }: ExportChatButtonProps) {
  const handleExport = useCallback(() => {
    const text = formatTranscript(messages);
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `chat-transcript-${new Date().toISOString().slice(0, 10)}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [messages]);

  if (messages.length === 0) return null;

  return (
    <Button
      size="icon"
      variant="ghost"
      title="Export chat transcript"
      aria-label="Export chat transcript"
      onClick={handleExport}
      className={className}
    >
      <Download className="size-4" />
    </Button>
  );
}
