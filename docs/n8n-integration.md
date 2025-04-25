# N8N Integration

This integration allows Open WebUI to communicate with workflows created in **n8n**, a powerful workflow automation tool. Messages are sent and received via webhook endpoints, making it easy to plug Open WebUI into your existing automation pipelines.

🔗 [Learn More About N8N](https://own.dev/n8n-io)


## Pipeline
- 🧩 [N8N Pipeline](https://own.dev/github-owndev-open-webui-functions-n8n-pipeline)


## Template Workflow

- 🧩 [N8N Open WebUI Test Agent (Template)](https://own.dev/github-owndev-open-webui-functions-open-webui-test-agent)


## Features

- **Webhook Communication**  
  Send messages directly to an n8n workflow via a webhook URL.

- **Token-Based Authentication**  
  Secure access to your n8n webhook using a Bearer token or Cloudflare Access tokens.

- **Flexible Input/Output Mapping**  
  Customize which fields in the request/response payload are used for communication.

- **Live Status Feedback**  
  Optionally emit status updates at a configurable interval.


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

# Interval (in seconds) between emitting status updates to the UI
EMIT_INTERVAL=2.0

# Enable or disable live status indicators
ENABLE_STATUS_INDICATOR=true

# Optional: Cloudflare Access tokens (if behind Cloudflare Zero Trust)
CF_ACCESS_CLIENT_ID="your-cloudflare-access-client-id"
CF_ACCESS_CLIENT_SECRET="your-cloudflare-access-client-secret"
```

> [!TIP]  
> If your n8n instance is protected behind Cloudflare Zero Trust, you can use service tokens for authentication.
> Learn more: [Cloudflare Access Service Tokens](https://own.dev/developers-cloudflare-com-cloudflare-one-identity-service-tokens)