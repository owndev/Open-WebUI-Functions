# Setup Azure Log Analytics Workspace
## Installation
- [Create Workspace](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/quick-create-workspace)

## Get Shared Key
To send data from Time Token Tracker to Azure Log Analytics, a shared key is required. This can be found under `Settings` > `Agents` > `Primary key`.

 <img src="./images/azure-log-analytics-get-key.png" />

## Show Logs
To view these logs, go to `Logs` > `Custom Logs`. All logs will be listed there.
> It may take a few minutes for the first logs to become visible.

 <img src="./images/azure-log-analytics-show-logs.png" />