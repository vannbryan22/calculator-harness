import type { Tool } from "./types.js";

export const readFileTool: Tool = {
  name: "read_file",
  description:
    'Reads the contents of a local file. Pass an object with a "filename" key. Example: {"filename": "secret_info.txt"}',
  execute: (args: Record<string, any>) => {
    try {
      const filename = args.filename;
      if (!filename) return "Error: Missing 'filename' argument.";

      // Bun has a built-in file system API!
      const file = Bun.file(filename);

      // Check if file exists
      if (!file.exists()) {
        return `Error: File '${filename}' not found.`;
      }

      // We use .text() to read the file asynchronously
      // Note: If you aren't using Bun, you'd use fs.readFileSync(filename, 'utf-8')
      return file.text();
    } catch (err: any) {
      return `Error reading file: ${err.message}`;
    }
  },
};
