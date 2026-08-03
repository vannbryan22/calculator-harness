

// --- HELPER: ESTIMATE TOKENS ---
// A simple rule of thumb: 1 token is about 4 characters.
export function estimateTokens(messages: Array<{ role: string; content: string }>): number {
  const totalChars = messages.map(m => m.content).join('').length;
  return Math.ceil(totalChars / 4);
}
