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

// --- COMPLETENESS RULES ---
// Each rule: if the user's question asks for a determination (the `ask` regex),
// the final answer MUST contain something matching the `need` regex.
// This guarantees multi-part questions aren't answered with only half the info
// (e.g. giving the result but forgetting to say "positive" or "negative").
type CompletenessRule = {
  ask: RegExp;
  need: RegExp;
  what: string;
  template: string;
};

const COMPLETENESS_RULES: CompletenessRule[] = [
  {
    ask: /\b(positive|negative)\b[\s\S]{0,60}\b(positive|negative)\b/i,
    need: /\b(positive|negative)\b/i,
    what: "whether the number is positive or negative",
    template:
      '"The exact result is [number], which is a [positive/negative] number."',
  },
  {
    ask: /\b(even|odd)\b[\s\S]{0,60}\b(even|odd)\b/i,
    need: /\b(even|odd)\b/i,
    what: "whether the number is even or odd",
    template: '"The result is [number], which is [even/odd]."',
  },
  {
    ask: /\b(true|false)\b[\s\S]{0,60}\b(true|false)\b/i,
    need: /\b(true|false)\b/i,
    what: "a true or false determination",
    template: '"[Statement] is [true/false]."',
  },
  {
    ask: /\b(yes|no)\b[\s\S]{0,60}\b(yes|no)\b/i,
    need: /\byes\b|\bno\b/i,
    what: "an explicit yes or no",
    template: '"[Yes/No], [brief explanation]."',
  },
];

// Returns the list of required determinations the candidate answer is missing.
function findMissingRequirements(
  question: string,
  answer: string,
): CompletenessRule[] {
  return COMPLETENESS_RULES.filter(
    (rule) => rule.ask.test(question) && !rule.need.test(answer),
  );
}

// Detect which tool a question needs, so we can directive-rescue a stuck model
// instead of asking it to "take an action" (which a small model can't do).
function inferNeededTool(
  userInput: string,
  usedTools: string[],
): { name: string; args: Record<string, string> } | null {
  const needsMath = /\b(subtract|add|multiply|divide|plus|minus|times|calculate|\d)\b/i.test(
    userInput,
  );
  const needsRead = /\b(read|open|view|show|contents? of)\b.{0,40}\b(file|\.txt|\.md|\.json|info|secret)\b/i.test(
    userInput,
  );
  const needsWrite =
    /\b(create|write|save)\b/i.test(userInput) &&
    /\b(file|\.txt|\.md|\.json)\b/i.test(userInput);

  // Order matters: read-before-math (you need the value first),
  // write last. Skip anything already used.
  if (needsRead && !usedTools.includes("read_file")) {
    const fileMatch = userInput.match(/\b([\w-]+\.(?:txt|md|json))\b/);
    const filename = fileMatch?.[1] ?? "";
    return {
      name: "read_file",
      args: filename ? { filename } : {},
    };
  }
  if (needsMath && !usedTools.includes("calculator")) {
    // Pull a "X - Y" style expression from the user text as a best effort.
    const exprMatch = userInput.match(
      /\b(\d[\d,\s]*)\s*(minus|subtract|-|less)\s*(\d[\d,\s]*)\b/i,
    );
    const expression =
      exprMatch && exprMatch[1] && exprMatch[3]
        ? `${exprMatch[1].replace(/\s|,/g, "")} - ${exprMatch[3].replace(/\s|,/g, "")}`
        : "";
    return {
      name: "calculator",
      args: expression ? { expression } : {},
    };
  }
  if (needsWrite && !usedTools.includes("write_file")) {
    return { name: "write_file", args: {} };
  }
  return null;
}

// Canonical key for a tool call's arguments: JSON with sorted top-level keys,
// so {"expression":"5*5"} and an equivalently-shaped re-call collide. Used to
// detect a tool call whose result is ALREADY KNOWN (don't redo known work).
function canonicalArgsKey(args: Record<string, any>): string {
  const sorted: Record<string, any> = {};
  for (const k of Object.keys(args).sort()) {
    sorted[k] = args[k];
  }
  return JSON.stringify(sorted);
}

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
     7. When writing your "critique", go through your CHECKLIST item by item. If the "draft_answer" does NOT address EVERY item on the checklist, you MUST say "REJECTED" and explain which item is missing. Only say "APPROVED" if every single checklist item is covered.

     GOOD EXAMPLE 1 (Before a tool):
     {"thought": "Checklist: 1. Do math 5*5. 2. Tell user the result. I need to use the calculator first.", "tool_name": "calculator", "tool_arguments": {"expression": "5 * 5"}, "draft_answer": null, "critique": null, "final_answer": null}

     GOOD EXAMPLE 2 (After a tool, but another tool is needed next):
     {"thought": "Checklist: 1. Read file. 2. Do math. I just read the file and got 100. Next step is to do math. I must call the calculator now.", "tool_name": "calculator", "tool_arguments": {"expression": "100 - 50"}, "draft_answer": null, "critique": null, "final_answer": null}

     GOOD EXAMPLE 3 (After all tools are done - Multi-part question):
     {"thought": "Checklist: 1. Subtract 50 from 100. 2. Tell if positive or negative. The calculator returned 50. 50 is greater than 0, so it is positive. I have answered both items.", "tool_name": null, "tool_arguments": {}, "draft_answer": "The result is 50, which is a positive number.", "critique": "APPROVED. The math is correct and I explicitly stated it is positive.", "final_answer": "The result is 50, which is a positive number."}
  `;

  messages = [{ role: "system", content: systemPrompt }];

  // Session-wide cache of tool results keyed by `${name}:${canonicalArgs}`.
  // Lets us detect a redundant tool call whose result is ALREADY KNOWN — for
  // example a follow-up turn where the model scrapes an old tool call from
  // conversation history and re-runs it. Scoped to the whole session so it
  // survives across turns; exact-args keyed so a genuinely new call is never
  // wrongly blocked.
  const seenToolResults = new Map<string, string>();

  while (true) {
    const userInput = prompt("You: ");
    const trimmed = (userInput ?? "").trim();

    // EOF (null) or explicit exit command → quit.
    if (userInput === null || /^(exit|quit|bye)$/i.test(trimmed)) {
      console.log("Goodbye!");
      break;
    }

    // Empty / whitespace line → re-prompt instead of quitting (the old
    // "!userInput" check treated an empty Enter as exit, causing "auto logout").
    if (trimmed === "") {
      console.log("(Type a question, or 'exit' to quit.)");
      continue;
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
    // Track which tools have actually executed for THIS user request, so we
    // can directive-rescue a stuck model (and not re-suggest a done tool).
    const usedTools: string[] = [];
    // Track the last assistant thought so we can detect a thought-only loop
    // (model emits only "thought" and repeats itself without ever acting).
    let lastThought = "";

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
            // --- REDUNDANT TOOL CALL GUARD (session-wide) ---
            // If we've ALREADY successfully run this exact (name, args) call
            // earlier in the session, do NOT re-run it. This is what stops a
            // small model from scraping an old tool call out of conversation
            // history and re-executing it on a follow-up turn (e.g. re-running
            // calculator("98765 - 100000") when its result is already known).
            // It also covers the bare-number no-op (calculator("-1235")).
            const callKey = `${selectedTool.name}:${canonicalArgsKey(parsed.tool_arguments)}`;
            const cached = seenToolResults.get(callKey);
            const isBareNumberCalc =
              selectedTool.name === "calculator" &&
              /^\s*-?\d+(\.\d+)?\s*$/.test(
                String(parsed.tool_arguments?.expression ?? ""),
              );

            if (cached !== undefined || isBareNumberCalc) {
              console.log(
                `❌ Harness Error: "${selectedTool.name}" was already called with these arguments${cached !== undefined ? ` (result: ${cached})` : ""}. Not re-running — answer the user's CURRENT message.`,
              );
              if (!usedTools.includes(selectedTool.name)) {
                usedTools.push(selectedTool.name);
              }
              messages.push({
                role: "user",
                content:
                  cached !== undefined
                    ? `You just called "${selectedTool.name}" again with arguments you've already used, getting the result: ${cached}. Re-running the SAME call will NOT help, and "${cached}" is NOT the answer to the user's latest message — do not echo it.

Look carefully at the user's LATEST message. It is asking for something NEW. If it requires NEW math, call the calculator with a DIFFERENT expression built from the user's new request (for example, if the user says "add X", use an expression like "${cached} + X"). Only if the message genuinely needs no new computation should you write a "draft_answer".

Output the JSON now.`
                    : `You called the calculator with a bare number (${String(parsed.tool_arguments?.expression ?? "")}), which has nothing to compute. Look at the user's LATEST message: if it needs math, call the calculator with a real EXPRESSION for it; otherwise write a "draft_answer". Output the JSON now.`,
              });
              continue;
            }

            console.log(
              `[Harness executing tool: ${selectedTool.name} with args: ${JSON.stringify(parsed.tool_arguments)}]`,
            );
            if (!usedTools.includes(selectedTool.name)) {
              usedTools.push(selectedTool.name);
            }
            try {
              const result = await selectedTool.execute(parsed.tool_arguments);
              console.log(`[Tool Result: ${result}]`);

              // Cache the successful result so a later identical call is caught.
              seenToolResults.set(callKey, String(result));

              // Directive nudge: if the user's request still needs a tool we
              // haven't run, point at it. Otherwise force a draft_answer now
              // (don't offer "a tool OR a draft" — a small model will re-call).
              const nextNeeded = inferNeededTool(userInput, usedTools);
              messages.push({
                role: "user",
                content: nextNeeded
                  ? `Tool Result: ${result}.\n\nNEXT STEP: Call the "${nextNeeded.name}" tool now${Object.keys(nextNeeded.args).length > 0 ? ` with arguments ${JSON.stringify(nextNeeded.args)}` : ""}. Output the JSON now.`
                  : `Tool Result: ${result}.\n\nYou now have all the information you need. STOP calling tools. You MUST write a "draft_answer" that answers the user's FULL question. Use this exact template:\n"The exact result is [insert the number from the tool result], which is a [positive/negative] number."\n\nOutput the JSON now with tool_name=null and your draft_answer.`,
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

        // ====================================================================
        // CANDIDATE EVALUATION (deterministic — the HARNESS decides, not the
        // model's self-critique). Keyed on `finalOutput`, NOT on `critique`,
        // so a draft with no critique is handled here (closes the dead zone
        // where nothing fired and the model got zero feedback).
        // ====================================================================
        if (finalOutput) {
          const isJustANumber = /^-?\d+(\.\d+)?$/.test(finalOutput.trim());
          const missing = findMissingRequirements(userInput, finalOutput);
          const lastAttempt = attempts >= maxAttempts;

          // Reject a bare number (unless it's the final attempt — accept best-effort).
          if (isJustANumber && !lastAttempt) {
            console.log(
              "❌ Harness Error: Model output a raw number instead of a full sentence!",
            );
            messages.push({
              role: "user",
              content: `Your draft answer was just a number: "${finalOutput}". You MUST write a full sentence. Use this exact template: "The exact result is [number], which is a [positive/negative] number." Output the JSON now.`,
            });
            continue;
          }

          // Reject an incomplete answer (unless it's the final attempt).
          if (missing.length > 0 && !lastAttempt) {
            const list = missing.map((m) => m.what).join(", ");
            const templates = missing.map((m) => m.template).join("\n");
            console.log(
              `❌ Harness Error: Answer is missing required info: ${list}`,
            );
            messages.push({
              role: "user",
              content: `Your answer does not address everything the user asked for. You are missing: ${list}.\n\nYou MUST rewrite your draft_answer to cover ALL of these. Use this template (filling in the placeholders):\n${templates}\n\nOutput the JSON with the corrected draft_answer now.`,
            });
            continue;
          }

          // ACCEPT. Best-effort warnings on the final attempt; otherwise normal.
          if (missing.length > 0) {
            console.log(
              "⚠️ Bypass: Accepting incomplete answer to prevent infinite loop.",
            );
          } else if (isJustANumber) {
            console.log(
              "⚠️ Bypass: Accepting raw number to prevent infinite loop.",
            );
          } else if (parsed.draft_answer) {
            console.log(`📝 Draft: ${parsed.draft_answer}`);
            if (parsed.critique) console.log(`🔍 Critique: ${parsed.critique}`);
          }

          console.log(`\nAgent: ${finalOutput}`);
          messages.push({ role: "assistant", content: finalOutput });
          isFinished = true;
          break;
        }

        // ====================================================================
        // NO CANDIDATE — directive rescue. Covers every non-tool response that
        // produced no answer: thought-only, critique-only, etc. Broadened from
        // "!draft && !critique && !tool" to "!finalOutput && !tool" so the dead
        // zone (e.g. critique-only) is also caught. Every path here continues.
        // ====================================================================
        if (!finalOutput && !parsed.tool_name) {
          const repeating =
            lastThought.length > 0 &&
            parsed.thought.trim() === lastThought.trim();
          lastThought = parsed.thought;

          // What's the next concrete step? inferNeededTool looks at the user's
          // request and which tools have already run THIS turn. When it returns
          // null there's no remaining tool to suggest — so we steer to a draft.
          // This is fully general: it handles greetings, follow-ups, and
          // "tools already done" through the same path, no word-matching.
          const needed = inferNeededTool(userInput, usedTools);

          // Escalate to a concrete skeleton only when the model is genuinely
          // stuck: repeating the same thought, or several attempts have failed.
          // NOTE: we deliberately do NOT escalate on attempt 1. An attempt-1
          // thought-only response is common and recoverable via the soft nudge
          // below, which lets the model reason freely about the CURRENT
          // question. Handing it a prescriptive skeleton too early biases a
          // small model toward copying old tool arguments from history.
          if (repeating || attempts >= 3) {
            // Escalation: a concrete skeleton the model can copy verbatim.
            let skeleton: string;
            if (needed) {
              const argHint =
                Object.keys(needed.args).length > 0
                  ? JSON.stringify(needed.args)
                  : `<fill in arguments for ${needed.name}>`;
              skeleton = `{"thought": "${parsed.thought.replace(/"/g, "'")} I will call the ${needed.name} tool now.", "tool_name": "${needed.name}", "tool_arguments": ${argHint}, "draft_answer": null, "critique": null, "final_answer": null}`;
            } else {
              // No tool left to suggest — write a draft answering the user's
              // latest message (covers conversational turns AND "all tools done").
              skeleton = `{"thought": "I will write my final answer now.", "tool_name": null, "tool_arguments": {}, "draft_answer": "<write a clear, full-sentence answer to the user's latest message here>", "critique": "APPROVED. The draft directly answers the user's latest message.", "final_answer": "<same as draft_answer>"}`;
            }
            console.log(
              "❌ Harness Error: Model produced no answer and is stuck. Sending a concrete JSON skeleton.",
            );
            messages.push({
              role: "user",
              content: `You did NOT set a "draft_answer"/"final_answer" (and called no tool) — so nothing happened. You are stuck.\n\nSTOP thinking. OUTPUT EXACTLY this JSON (fill in any [bracketed] placeholders with real values, then send it):\n\n${skeleton}`,
            });
          } else {
            // First/second miss: be directive about the exact next step.
            console.log("❌ Harness Error: Model didn't take any action!");
            const directive = needed
              ? `You MUST call the "${needed.name}" tool right now${
                  Object.keys(needed.args).length > 0
                    ? ` with these arguments: ${JSON.stringify(needed.args)}`
                    : ""
                }.`
              : `You have no remaining tool to call. You MUST write a "draft_answer" that directly answers the user's latest message.`;
            messages.push({
              role: "user",
              content: `You did not provide a tool_name, draft_answer, or final_answer. ${directive}\n\nRemember the required JSON shape:\n{"thought": "...", "tool_name": "..." or null, "tool_arguments": {...}, "draft_answer": "..." or null, "critique": "..." or null, "final_answer": "..." or null}\n\nOutput the JSON now.`,
            });
          }
          continue;
        }

        // Reached only with a tool_name but no candidate and no rescue hit —
        // (e.g. unknown tool handled above already continues). Reset loop detector.
        lastThought = "";
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
