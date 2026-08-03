// helpers/summarize-history.ts
import { ollama } from "../index.js";
import type { Message } from "../types.ts";

export async function summarizeHistory(
  oldMessages: Array<Message>,
): Promise<string> {
  const summaryPrompt = [
    {
      role: "system",
      content:
        "You are a memory management AI. Summarize the following conversation history into a concise paragraph. Focus on the user's goal, the tools used, and the results. Output ONLY a plain string, no JSON.",
    },
    ...oldMessages,
  ];

  const response = await ollama.chat({
    model: "qwen2.5vl:3b",
    messages: summaryPrompt as any,
    stream: false,
  });

  const content = response.message.content || "";

  // If it accidentally outputs JSON, extract the text
  if (content.startsWith("{")) {
    try {
      const parsed = JSON.parse(content);
      return (
        parsed.thought || parsed.final_answer || parsed.draft_answer || content
      );
    } catch {
      return content;
    }
  }
  return content;
}
