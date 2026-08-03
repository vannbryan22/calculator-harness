import type { Tool } from "./types.js";

export const timeTool: Tool = {
  name: 'get_current_time',
  description: 'Useful for finding out what time it is right now. No arguments needed, pass an empty object {}.',
  execute: (args: Record<string, any>) => {
    return new Date().toLocaleString();
  }
};
