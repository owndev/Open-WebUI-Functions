# Infomaniak AI Tools Integration

This integration enables Open WebUI to interact with **Infomaniak AI Tools**, using their public API for secure and scalable AI model access.

🔗 [Learn More About N8N](https://own.dev/n8n-io)


## Pipeline

- 🧩 [Infomaniak AI Tools Pipeline](https://own.dev/github-owndev-open-webui-functions-infomaniak)


## Features

- **Secure API Access**  
  Authenticate via API key (automatically encrypted and stored securely).

- **Model Product Binding**  
  Associate API requests with a specific product ID provided by Infomaniak.

- **Customizable API Endpoint**  
  Define a custom base URL for regional or private deployments.

- **Model Name Prefixing**  
  Automatically add a prefix to distinguish models from other providers.


## Environment Variables

Set the following environment variables to enable Infomaniak AI Tools integration:

```bash
# API key for authenticating with Infomaniak AI Tools
INFOMANIAK_API_KEY="your-api-key"

# Product ID (default: 50070) assigned by Infomaniak
INFOMANIAK_PRODUCT_ID=50070

# Base URL of the Infomaniak API
INFOMANIAK_BASE_URL="https://api.infomaniak.com"

# Optional: Prefix to add before model names (e.g. for display or routing)
NAME_PREFIX="Infomaniak: "
```

> [!TIP]  
> You can find your API key and product ID in your [Infomaniak Manager](https://own.dev/manager-infomaniak-com).