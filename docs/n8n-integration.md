# N8N Integration

This integration allows Open WebUI to communicate with workflows created in **n8n**, a powerful workflow automation tool. Messages are sent and received via webhook endpoints, making it easy to plug Open WebUI into your existing automation pipelines.

> [!NOTE]  
> **Recent Improvements (v2.2.0)**: Added AI Agent tool usage display with collapsible details sections. The pipeline now extracts and displays tool calls from `intermediateSteps` in **non-streaming mode**, showing tool names, inputs, and results in a user-friendly format.

> [!IMPORTANT]  
> **Tool Usage Display**: AI Agent tool calls are currently only visible in **non-streaming mode** due to N8N's streaming implementation. The code is future-proof and will automatically work if N8N adds `intermediateSteps` support to streaming responses.

🔗 [Learn More About N8N](https://n8n.io/)

## Pipeline

- 🧩 [N8N Pipeline](../pipelines/n8n/n8n.py)

## Template Workflow

- 🧩 [N8N Open WebUI Test Agent (Streaming)](../pipelines/n8n/Open_WebUI_Test_Agent_Streaming.json)
- 🧩 [N8N Open WebUI Test Agent (Non-Streaming)](../pipelines/n8n/Open_WebUI_Test_Agent.json)

## Features

> [!TIP]
> **AI Agent Tool Usage Display (v2.2.0)** 🛠️
>
> Automatically extracts and displays tool calls from N8N AI Agent workflows in **non-streaming mode**.
>
> - 📊 **Three verbosity levels**: `minimal` (names only), `compact` (names + preview), `detailed` (full info)
> - 📏 **Customizable length limits**: Control input/output text length
> - 🎯 **Flexible configuration**: Adapt display to your needs
>
> Shows tool names, inputs, and results from the `intermediateSteps` array to provide transparency into the AI agent's workflow execution.
>
> 📖 [Learn more about Tool Usage Display](./n8n-tool-usage-display.md)

- **Streaming & Non-Streaming Support**  
  Automatic detection and handling of both streaming and non-streaming responses with consistent output formatting.

- **SystemPrompt Deduplication**  
  Intelligent removal of duplicate system prompts to prevent redundant instructions.

- **Webhook Communication**  
  Send messages directly to an n8n workflow via a webhook URL.

- **Token-Based Authentication**  
  Secure access to your n8n webhook using a Bearer token or Cloudflare Access tokens.

- **Flexible Input/Output Mapping**  
  Customize which fields in the request/response payload are used for communication.

- **Robust Error Handling**  
  All errors are displayed as user-friendly English messages in the chat interface.

## Environment Variables

Set the following environment variables to enable n8n integration:

```bash
# n8n webhook endpoint
# Example: "https://n8n.yourdomain.com/webhook/openwebui-agent"
N8N_URL="https://<your-endpoint>/webhook/<your-webhook>"

# Optional: Bearer token for secure access
N8N_BEARER_TOKEN="your-bearer-token"

# Payload input field (used by Open WebUI to send messages)
INPUT_FIELD="chatInput"

# Payload output field (used by Open WebUI to read the response)
RESPONSE_FIELD="output"

# Optional: Cloudflare Access tokens (if behind Cloudflare Zero Trust)
CF_ACCESS_CLIENT_ID="your-cloudflare-access-client-id"
CF_ACCESS_CLIENT_SECRET="your-cloudflare-access-client-secret"
```

> [!TIP]  
> If your n8n instance is protected behind Cloudflare Zero Trust, you can use service tokens for authentication.
> Learn more: [Cloudflare Access Service Tokens](https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/)
