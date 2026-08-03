import type { Tool } from "./types.js";

export const writeFileTool: Tool = {
  name: "write_file",
  description:
    'Writes content to a local file. Pass an object with "filename" and "content" keys. Example: {"filename": "notes.txt", "content": "Hello World"}',
  execute: async (args: Record<string, any>) => {
    try {
      const filename = args.filename;
      const content = args.content;

      if (!filename || content === undefined) {
        return "Error: Missing 'filename' or 'content' argument.";
      }

      // Bun's native file writing API
      await Bun.write(filename, content);

      return `Success: File '${filename}' was written successfully.`;
    } catch (err: any) {
      return `Error writing file: ${err.message}`;
    }
  },
};
