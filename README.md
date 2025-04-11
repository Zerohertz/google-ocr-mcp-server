<h2 align = "center">
    📸 Google OCR MCP server 📸
</h2>

<p align="center">
    </br>
    <a href="https://github.com/Zerohertz/google-ocr-mcp-server">
        <img src="https://img.shields.io/badge/GitHub-181717?&logo=GitHub&logoColor=white" />
    </a>
    <a href="https://smithery.ai/server/@Zerohertz/google-ocr-mcp-server">
        <img src="https://smithery.ai/badge/@Zerohertz/google-ocr-mcp-server" />
    </a>
    <a href="https://pypi.org/project/google-ocr-mcp-server/">
        <img src="https://img.shields.io/pypi/v/google-ocr-mcp-server?&logo=PyPI&logoColor=FFFFFF&labelColor=3775A9&color=007EC6" />
    </a>
    <a href="https://github.com/Zerohertz/google-ocr-mcp-server/blob/master/LICENSE">
        <img src="https://img.shields.io/pypi/l/google-ocr-mcp-server" />
    </a>
    <a href="https://pypi.org/project/google-ocr-mcp-server/">
        <img src="https://img.shields.io/pypi/wheel/google-ocr-mcp-server" />
    </a>
</p>

## Components

### Resources

The server implements a simple note storage system with:

- Custom note:// URI scheme for accessing individual notes
- Each note resource has a name, description and text/plain mimetype

### Prompts

The server provides a single prompt:

- summarize-notes: Creates summaries of all stored notes
  - Optional "style" argument to control detail level (brief/detailed)
  - Generates prompt combining all current notes with style preference

### Tools

The server implements one tool:

- add-note: Adds a new note to the server
  - Takes "name" and "content" as required string arguments
  - Updates server state and notifies clients of resource changes

## Configuration

[TODO: Add configuration details specific to your implementation]

## Quickstart

### Install

#### Claude Desktop

- On MacOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
- On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

<details>
  <summary>Development/Unpublished Servers Configuration</summary>

```json
{
  "mcpServers": {
    "google-ocr-mcp-server": {
      "command": "uv",
      "args": ["run", "google-ocr-mcp-server"],
      "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/google-application-credentials.json",
        "SAVE_RESULTS": false
      }
    }
  }
}
```

</details>

<details>
  <summary>Published Servers Configuration</summary>

```json
{
  "mcpServers": {
    "google-ocr-mcp-server": {
      "command": "uvx",
      "args": ["google-ocr-mcp-server"],
      "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/google-application-credentials.json",
        "SAVE_RESULTS": false
      }
    }
  }
}
```

</details>

### Installing via Smithery

To install google-ocr-mcp-server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@Zerohertz/google-ocr-mcp-server):

```bash
npx -y @smithery/cli install @Zerohertz/google-ocr-mcp-server --client claude
```

## Development

### Building and Publishing

To prepare the package for distribution:

1. Sync dependencies and update lockfile:

```bash
uv sync
```

2. Build package distributions:

```bash
uv build
```

This will create source and wheel distributions in the `dist/` directory.

3. Publish to PyPI:

```bash
uv publish
```

Note: You'll need to set PyPI credentials via environment variables or command flags:

- Token: `--token` or `UV_PUBLISH_TOKEN`
- Or username/password: `--username`/`UV_PUBLISH_USERNAME` and `--password`/`UV_PUBLISH_PASSWORD`

### Debugging

Since MCP servers run over stdio, debugging can be challenging. For the best debugging
experience, we strongly recommend using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector).

You can launch the MCP Inspector via [`npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) with this command:

```bash
$ npx @modelcontextprotocol/inspector uv --directory /Users/zerohertz/Downloads/google-ocr-mcp-server run google-ocr-mcp-server
Starting MCP inspector...
⚙️ Proxy server listening on port 6277
🔍 MCP Inspector is up and running at http://127.0.0.1:6274 🚀
```

Upon launching, the Inspector will display a URL that you can access in your browser to begin debugging.
