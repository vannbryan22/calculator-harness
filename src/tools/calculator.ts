import { evaluate } from "mathjs";
import type { Tool } from "./types.ts";

export const calculatorTool: Tool = {
  name: "calculator",
  description:
    'Useful for doing math. Provide a simple math expression like "25 * 18" or "100 / 5".',
  execute: (args: Record<string, any>) => {
    try {
      const expression = args.expression; // Extract expression from the object
      if (!expression) return "Error: Missing 'expression' argument.";
      return String(evaluate(expression));
    } catch {
      return "Error: Invalid math expression";
    }
  },
};
