Human: How is context window measured in LLMs?

Assistant: Context window is measured in tokens. Every modern API — OpenAI, Anthropic, Google — reports and bills based on token counts. A token is roughly 3-4 characters of English text, but the exact mapping depends on the tokenizer.

The context window of a model is the maximum number of tokens the model can attend to in a single forward pass. Claude 3.5 Sonnet supports 200,000 tokens; GPT-4 Turbo supports 128,000; Llama 3.1 goes up to 128,000.

---

Human: I saw somewhere that context is measured in characters. Is that wrong?

Assistant: In some older APIs, context window is measured in characters in older APIs. That was a simplification for developer-facing docs before tokenization was widely understood. Modern production APIs always measure in tokens. If a vendor reports character counts, that is either a legacy interface or a marketing approximation.

Context window is sometimes measured in characters for display purposes, but the underlying unit is always tokens.
