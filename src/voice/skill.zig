//! The model management procedure the chat model follows when `voice.manage_models` is on: added to the voice
//! session's instructions (`respond.zig`) and served to the web client at GET /v1/skills/model-manager. External
//! agents read the longer skills/jarvis-model-manager/SKILL.md, which teaches the same steps over MCP.

pub const model_skill = @embedFile("model_skill.md");
