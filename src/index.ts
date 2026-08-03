import { Ollama } from "ollama";
import { z } from "zod";
import { estimateTokens } from "./helpers/estimate-tokens.js";
import { summarizeHistory } from "./helpers/summarize-history.js";
import type { Message } from "./types.ts";
import type { Tool } from "./tools/types.js";
import { calculatorTool } from "./tools/calculator.js";
import { timeTool } from "./tools/time.js";
import { extractJson } from "./helpers/extract-json.js";
import { readFileTool } from "./tools/read-file.js";
import { AgentSchema } from "./schemas.js";
import { writeFileTool } from "./tools/write-file.js";

export const ollama = new Ollama({ host: "http://localhost:11434" });

const toolRegistry: Tool[] = [
  calculatorTool,
  timeTool,
  readFileTool,
  writeFileTool,
];

let messages: Array<Message> = [];
type AgentResponse = z.infer<typeof AgentSchema>;

async function runHarness() {
  let toolDescriptions = toolRegistry
    .map((tool) => `- ${tool.name}: ${tool.description}`)
    .join("\n");

  const systemPrompt = `
     You are a helpful AI assistant.
     You MUST respond with ONLY valid JSON. Do not include markdown formatting or conversational text.

     Available Tools:
     ${toolDescriptions}

     You must output EXACTLY this JSON structure:
     {
       "thought": "A single string containing your step-by-step reasoning and checklist. Do NOT use an array.",
       "tool_name": "The name of the tool, or null if you have the final answer.",
       "tool_arguments": "A JSON object of arguments for the tool, or an empty object {} if no tool is used.",
       "draft_answer": "A draft answer based on the tool results, or null if using a tool.",
       "critique": "Review your draft. Say 'APPROVED' if correct, or 'REJECTED' if flawed.",
       "final_answer": "The final answer if approved, or null if rejected/using a tool."
     }

     CRITICAL RULES:
     1. "final_answer" and "draft_answer" MUST be full, human-readable sentences. NEVER output a raw number.
     2. If you need to use a tool, "draft_answer", "critique", and "final_answer" MUST be null.
     3. After a tool has run, you MUST fill out "draft_answer" and "critique". Do not skip these steps.
     4. In your "thought" field, create a CHECKLIST of everything the user asked for.
     5. You are terrible at math. If the user asks you to subtract, add, multiply, or divide, you MUST use the calculator tool.
     6. Treat every new user message as a fresh task. Do not repeat previous final answers.

     GOOD EXAMPLE 1 (Before a tool):
     {"thought": "Checklist: 1. Do math 5*5. 2. Tell user the result. I need to use the calculator first.", "tool_name": "calculator", "tool_arguments": {"expression": "5 * 5"}, "draft_answer": null, "critique": null, "final_answer": null}

     GOOD EXAMPLE 2 (After a tool, but another tool is needed next):
     {"thought": "Checklist: 1. Read file. 2. Do math. I just read the file and got 100. Next step is to do math. I must call the calculator now.", "tool_name": "calculator", "tool_arguments": {"expression": "100 - 50"}, "draft_answer": null, "critique": null, "final_answer": null}

     GOOD EXAMPLE 3 (After all tools are done - Multi-part question):
     {"thought": "Checklist: 1. Subtract 50 from 100. 2. Tell if positive or negative. The calculator returned 50. 50 is greater than 0, so it is positive. I have answered both items.", "tool_name": null, "tool_arguments": {}, "draft_answer": "The result is 50, which is a positive number.", "critique": "APPROVED. The math is correct and I explicitly stated it is positive.", "final_answer": "The result is 50, which is a positive number."}
  `;

  messages = [{ role: "system", content: systemPrompt }];

  while (true) {
    const userInput = prompt("You: ");

    if (!userInput || userInput.toLowerCase() === "exit") {
      console.log("Goodbye!");
      break;
    }

    messages.push({ role: "user", content: userInput });

    // CRITICAL: Smack the model awake so it doesn't repeat old answers!
    messages.push({
      role: "system",
      content: `ALERT: The user just sent a NEW message: "${userInput}". You MUST evaluate this new message. If it is a greeting (like "hey"), respond with a greeting. If it is a follow-up question, answer it. DO NOT repeat your previous JSON or your previous final answer. Start a fresh thought process.`,
    });

    let attempts = 0;
    const maxAttempts = 6;
    let isFinished = false;

    while (attempts < maxAttempts) {
      attempts++;
      console.log(`\n--- Attempt ${attempts} ---`);

      // --- MEMORY MANAGEMENT ---
      const tokenEstimate = estimateTokens(messages);
      console.log(`[Context size: ~${tokenEstimate} tokens]`);

      if (tokenEstimate > 3000) {
        console.log("🧠 Context getting long. Summarizing older messages...");
        const systemPromptMsg = messages[0];
        const recentMessages = messages.slice(-4);
        const oldMessages = messages.slice(1, -4);

        if (oldMessages.length > 0) {
          const summary = await summarizeHistory(oldMessages);
          console.log(`[Summary created: ${summary.substring(0, 50)}...]`);
          messages = [
            systemPromptMsg as Message,
            {
              role: "system",
              content: `Summary of previous conversation: ${summary}`,
            },
            ...recentMessages,
          ];
        }
      }

      const response = await ollama.chat({
        model: "qwen2.5vl:3b",
        messages: messages as any,
        format: "json",
        stream: true,
      });

      let rawReply = "";
      process.stdout.write("🤖 Thinking ");
      for await (const chunk of response) {
        process.stdout.write(".");
        rawReply += chunk.message.content;
      }
      console.log(" Done!\n");

      messages.push({ role: "assistant", content: rawReply });
      const jsonString = extractJson(rawReply);

      if (!jsonString) {
        messages.push({
          role: "user",
          content: "You did not output JSON. Please output ONLY valid JSON.",
        });
        continue;
      }

      try {
        const parsed: AgentResponse = AgentSchema.parse(JSON.parse(jsonString));
        console.log("✅ Zod Validation Passed!");
        console.log(`Thought: ${parsed.thought}`);

        // --- 1. DYNAMIC TOOL EXECUTION ---
        if (parsed.tool_name) {
          const selectedTool = toolRegistry.find(
            (t) => t.name === parsed.tool_name,
          );

          if (selectedTool) {
            console.log(
              `[Harness executing tool: ${selectedTool.name} with args: ${JSON.stringify(parsed.tool_arguments)}]`,
            );
            try {
              const result = await selectedTool.execute(parsed.tool_arguments);
              console.log(`[Tool Result: ${result}]`);

              messages.push({
                role: "user",
                content: `Tool Result: ${result}. \n\nCRITICAL INSTRUCTION: Based on this tool result, you MUST now either provide a "tool_name" for the next tool, OR provide a "draft_answer". Do not leave them null. Output the JSON now.`,
              });
            } catch (err: any) {
              console.log(`[Tool Error: ${err.message}]`);
              messages.push({
                role: "user",
                content: `Tool Error: The tool failed with the message: ${err.message}. Please try again or use a different tool.`,
              });
            }
            continue;
          } else {
            messages.push({
              role: "user",
              content: `Tool "${parsed.tool_name}" does not exist. Available tools: ${toolRegistry.map((t) => t.name).join(", ")}`,
            });
            continue;
          }
        }

        // --- 2. MATH ENFORCER ---
        if (
          !parsed.tool_name &&
          userInput.match(
            /\b(subtract|add|multiply|divide|plus|minus|times)\b/i,
          )
        ) {
          const hasUsedCalculator = messages.some(
            (m) => m.role === "assistant" && m.content.includes('"calculator"'),
          );
          if (
            !hasUsedCalculator &&
            (parsed.draft_answer || parsed.final_answer)
          ) {
            console.log(
              "❌ Harness Error: Model tried to do math in its head instead of using the calculator!",
            );
            messages.push({
              role: "user",
              content:
                "You tried to do math in your head! Reading a file is not enough. You MUST use the 'calculator' tool. Output the JSON to call the calculator tool now.",
            });
            continue;
          }
        }

        // --- 3. WRITE FILE ENFORCER ---
        if (
          userInput.toLowerCase().match(/\b(create|write|save)\b/) &&
          userInput.toLowerCase().match(/\b(file|\.txt|\.md|\.json)\b/)
        ) {
          const hasUsedWriteFile = messages.some(
            (m) => m.role === "assistant" && m.content.includes('"write_file"'),
          );

          if (
            !hasUsedWriteFile &&
            (parsed.draft_answer || parsed.final_answer)
          ) {
            console.log(
              "❌ Harness Error: Model tried to answer without writing the file!",
            );
            messages.push({
              role: "user",
              content:
                "You tried to answer the user, but you forgot to actually CREATE the file! You MUST use the 'write_file' tool to write the content to the file. Output the JSON to call the write_file tool now.",
            });
            continue;
          }
        }

        let finalOutput = parsed.final_answer || parsed.draft_answer || "";

        // --- 4. RAW NUMBER ENFORCER ---
        if (finalOutput) {
          const isJustANumber = /^-?\d+(\.\d+)?$/.test(finalOutput.trim());
          if (isJustANumber && attempts < 5) {
            console.log(
              "❌ Harness Error: Model output a raw number instead of a full sentence!",
            );
            messages.push({
              role: "user",
              content: `Your draft answer was just a number: "${finalOutput}". You MUST write a full sentence. Use this exact template: "The exact result is [number], which is a [positive/negative] number." Output the JSON now.`,
            });
            continue;
          }
        }

        // --- 5. CRITIC REJECTION CHECK ---
        if (parsed.draft_answer && parsed.critique && !parsed.final_answer) {
          console.log(`📝 Draft: ${parsed.draft_answer}`);
          console.log(`🔍 Critique: ${parsed.critique}`);

          if (!parsed.critique.includes("APPROVED")) {
            console.log(
              "❌ Harness Error: Critique did not approve the draft!",
            );
            messages.push({
              role: "user",
              content: `Your critique did not say 'APPROVED'. You must fix your draft.\n\nCRITICAL INSTRUCTION: Do not output a raw number. Use this exact template for your draft_answer:\n"The exact result is [insert number], which is a [positive/negative] number."\n\nOutput the JSON with the corrected draft_answer now.`,
            });
            continue;
          }
        }

        // --- 6. FINAL APPROVAL & OUTPUT ---
        if (
          parsed.critique &&
          parsed.critique.includes("APPROVED") &&
          finalOutput
        ) {
          if (attempts >= 5 && /^-?\d+(\.\d+)?$/.test(finalOutput.trim())) {
            console.log(
              "⚠️ Bypass: Accepting raw number to prevent infinite loop.",
            );
          } else {
            console.log(`📝 Draft: ${parsed.draft_answer}`);
            console.log(`🔍 Critique: ${parsed.critique}`);
          }

          console.log(`\nAgent: ${finalOutput}`);
          messages.push({ role: "assistant", content: finalOutput });
          isFinished = true;
          break;
        }

        // --- 7. ACTION ENFORCER ---
        if (!parsed.draft_answer && !parsed.critique && !parsed.tool_name) {
          console.log("❌ Harness Error: Model didn't take any action!");
          messages.push({
            role: "user",
            content: `You did not provide a tool_name, draft_answer, or final_answer. You MUST take an action. If you are not done, call a tool. If you have all the information, write a draft_answer. Output the JSON now.`,
          });
        }
      } catch (error) {
        if (error instanceof z.ZodError) {
          console.log("❌ Zod Validation Failed:");
          const errorMessages = error.issues
            .map((e) => `Path: ${e.path.join(".")}, Issue: ${e.message}`)
            .join("\n");
          console.log(errorMessages);
          messages.push({
            role: "user",
            content: `Your JSON was invalid. Zod Errors:\n${errorMessages}\n\nPlease fix the JSON and output it again.`,
          });
        } else {
          console.log("JSON Parse error:", error);
          messages.push({
            role: "user",
            content:
              "Your JSON was malformed. Please output strictly valid JSON.",
          });
        }
      }
    }

    if (!isFinished) {
      console.log(
        "\nAgent: I'm sorry, I got stuck and couldn't figure that out. Let's try something else.",
      );
    }
  }
}

runHarness();
