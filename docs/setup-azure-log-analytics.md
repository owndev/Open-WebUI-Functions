# Setup Azure Log Analytics Workspace

> [!WARNING]
> The Time Token Tracker sends its records through the Azure Monitor [HTTP Data Collector API](https://learn.microsoft.com/en-us/previous-versions/azure/azure-monitor/logs/data-collector-api) (`https://<workspace-id>.ods.opinsights.azure.com/api/logs`, signed with the workspace shared key). Microsoft has deprecated this API. Support ended on **September 14, 2026**, but ingestion still works for clients that use TLS 1.2 or later. The API does not work with workspaces behind an Azure Monitor Private Link Scope (AMPLS). Microsoft recommends the [Logs ingestion API](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/custom-logs-migrate) instead, which the filter does not support yet.

## Installation

- [Create Workspace](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/quick-create-workspace)

## Get Shared Key

To send data from Time Token Tracker to Azure Log Analytics, a shared key is required. This can be found under `Settings` > `Agents` > `Primary key`.

 <img src="./images/azure-log-analytics-get-key.png" />

## Configure the Time Token Tracker

Set the valves of the Time Token Tracker filter (**Admin Panel → Functions → Time Token Tracker → Valves**). You can also set defaults with environment variables of the Open WebUI container. Values saved in the valves take precedence over these.

| Valve / environment variable | Description |
| --- | --- |
| `SEND_TO_LOG_ANALYTICS` | `true` turns on sending (`1`, `yes` and `on` also work). Any other value, including `false`, turns it off. Default: off. |
| `LOG_ANALYTICS_WORKSPACE_ID` | The Workspace ID. |
| `LOG_ANALYTICS_SHARED_KEY` | The primary key from above. When saved as a valve, it is stored encrypted if `WEBUI_SECRET_KEY` is set. |
| `LOG_ANALYTICS_LOG_TYPE` | Valve only. The name of the custom log. Default: `OpenWebuiMetrics`. |

When sending is on, each response creates one record with `timestamp` (UTC, ISO 8601 with `Z`), `chatId`, `messageId`, `model`, `userId`, `responseTime`, `requestTokens`, `responseTokens`, `tokensPerSecond` and `tokensEstimated`. With `CALCULATE_ALL_MESSAGES` and `SHOW_AVERAGE_TOKENS` on (the default), it also has `avgRequestTokens` and `avgResponseTokens`. Log Analytics adds a type suffix to the names of custom fields, for example `requestTokens_d` and `tokensEstimated_b`.

The filter sends the record in the background, so the response does not wait for Log Analytics. A send that takes longer than 10 seconds, or more than 5 seconds to connect, is aborted. Failed sends are logged as errors in the Open WebUI log and are not retried.

> [!NOTE]
> Since Open WebUI 0.10, outlet filters also run for requests sent directly to the API (`/api/chat/completions` without a chat). Since version 2.6.2 the filter records these requests as well. They do not belong to a chat, so their `chatId` is a generated UUID.

How the values are measured:

- `requestTokens` counts the messages as the filter's inlet step sees them, before Open WebUI adds RAG context, web search results or a code interpreter prompt.
- While no `tiktoken` encoding is loaded (for example offline, without a `tiktoken` cache), token counts are estimates (about 4 characters per token). Such records have `tokensEstimated` set to `true`, and the status message in the chat shows `~` before the estimated numbers.
- The filter matches each response to its request through the request metadata that Open WebUI 0.11 passes to both filter steps. If the outlet step gets other metadata (for example through the legacy `/api/chat/completed` endpoint), it matches by user, model and last user message instead. If no request matches, the record has `responseTime` and `requestTokens` 0 and a warning is logged.

## Show Logs

To view these logs, go to `Logs` > `Custom Logs`. All logs will be listed there.
> It may take a few minutes for the first logs to become visible.

 <img src="./images/azure-log-analytics-show-logs.png" />

## PowerBI Dashboard

- [PowerBI Dashboard](https://github.com/owndev/Open-WebUI-Functions/discussions/26) from [@zic04](https://github.com/zic04)
