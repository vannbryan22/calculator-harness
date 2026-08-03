import { z } from "zod";

export const AgentSchema = z.object({
  // Accept string, array, or null. Default to empty string.
  thought: z
    .union([z.string(), z.array(z.string()), z.null()])
    .transform((val) => (Array.isArray(val) ? val.join(" ") : val || ""))
    .describe("Your step-by-step reasoning."),

  // Accept string or null. Default to null.
  tool_name: z
    .string()
    .nullable()
    .default(null)
    .describe("The exact name of the tool..."),

  // Accept record, null, or undefined. Default to empty object.
  tool_arguments: z
    .record(z.any(), z.any())
    .default({})
    .describe("A JSON object of arguments..."),

  // Accept string or null. Default to null.
  draft_answer: z
    .string()
    .nullable()
    .default(null)
    .describe("A draft of your final answer..."),

  critique: z
    .string()
    .nullable()
    .default(null)
    .describe("Review your draft..."),

  final_answer: z
    .string()
    .nullable()
    .default(null)
    .describe("The final, polished answer..."),
});
