# Setup Azure Log Analytics Workspace

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

When sending is on, each response creates one record with `chatId`, `messageId`, `model`, `userId`, `responseTime`, `requestTokens`, `responseTokens` and `tokensPerSecond`. With `CALCULATE_ALL_MESSAGES` and `SHOW_AVERAGE_TOKENS` on (the default), it also has `avgRequestTokens` and `avgResponseTokens`.

> [!NOTE]
> Since Open WebUI 0.10, outlet filters also run for requests sent directly to the API (`/api/chat/completions` without a chat). Since version 2.6.2 the filter records these requests as well. They do not belong to a chat, so their `chatId` is a generated UUID.

## Show Logs

To view these logs, go to `Logs` > `Custom Logs`. All logs will be listed there.
> It may take a few minutes for the first logs to become visible.

 <img src="./images/azure-log-analytics-show-logs.png" />

## PowerBI Dashboard

- [PowerBI Dashboard](https://github.com/owndev/Open-WebUI-Functions/discussions/26) from [@zic04](https://github.com/zic04)
