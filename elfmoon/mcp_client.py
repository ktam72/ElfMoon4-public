"""MCP クライアント — MCP サーバとの接続・ツール一覧・ツール呼び出しを管理する。

設定ファイルは ELFMOON_MCP_CONFIG 環境変数で指定。
未指定の場合、以下の順で自動検出:
  1. ~/.config/opencode/opencode.json (opencode 書式)
  2. ~/.config/opencode/mcp.json (標準 MCP 書式)

対応書式:
  opencode 書式:
    { "mcp": { "filesystem": { "type":"local", "command":["npx","-y","server"], "enabled":true } } }
  標準 MCP 書式:
    { "mcpServers": { "filesystem": { "command":"npx", "args":["-y","server"], "env":{} } } }
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_OPENCODE_CONFIG = os.path.expanduser("~/.config/opencode/opencode.json")
_STANDARD_CONFIG = os.path.expanduser("~/.config/opencode/mcp.json")


def _discover_config() -> str | None:
    """設定ファイルを自動検出する。"""
    env = os.environ.get("ELFMOON_MCP_CONFIG")
    if env:
        return env if os.path.exists(env) else None
    if os.path.exists(_OPENCODE_CONFIG):
        return _OPENCODE_CONFIG
    if os.path.exists(_STANDARD_CONFIG):
        return _STANDARD_CONFIG
    return None


def _normalize_servers(config: dict) -> dict[str, dict]:
    """opencode 書式と標準 MCP 書式の両方を正規化する。"""
    servers = {}

    # opencode 書式: { "mcp": { "name": { "command": [...], "enabled": true } } }
    mcp_block = config.get("mcp") or config.get("mcpServers", {})
    if not isinstance(mcp_block, dict):
        return servers

    for name, srv in mcp_block.items():
        if isinstance(srv, dict) and srv.get("enabled") is False:
            continue
        if not isinstance(srv, dict):
            continue

        if "command" in srv:
            cmd = srv["command"]
            if isinstance(cmd, list):
                # opencode 書式: command がリスト
                servers[name] = {
                    "command": cmd[0],
                    "args": cmd[1:],
                    "env": srv.get("env", {}),
                }
            elif isinstance(cmd, str):
                # 標準 MCP 書式: command が文字列 + args がリスト
                servers[name] = {
                    "command": cmd,
                    "args": srv.get("args", []),
                    "env": {**os.environ, **srv.get("env", {})},
                }
    return servers


class MCPError(Exception):
    pass


class MCPClientManager:
    def __init__(self):
        self._server_configs: dict[str, dict] = {}
        self._openai_tools: list[dict] = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        path = _discover_config()
        if not path:
            print("[MCP] 設定ファイルなし", file=sys.stderr, flush=True)
            self._loaded = True
            return
        with open(path) as f:
            config = json.load(f)
        self._server_configs = _normalize_servers(config)
        if not self._server_configs:
            print("[MCP] サーバ定義なし", file=sys.stderr, flush=True)
            self._loaded = True
            return
        print(
            f"[MCP] {path}: {len(self._server_configs)} サーバ定義",
            file=sys.stderr,
            flush=True,
        )
        self._refresh_tools()
        self._loaded = True

    def _refresh_tools(self):
        async def _list_all():
            tools = []
            for name, srv in self._server_configs.items():
                try:
                    params = StdioServerParameters(
                        command=srv["command"],
                        args=srv.get("args", []),
                        env={**os.environ, **srv.get("env", {})},
                    )
                    async with stdio_client(params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.list_tools()
                            for t in result.tools:
                                tools.append(
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": f"{name}__{t.name}",
                                            "description": t.description or "",
                                            "parameters": t.inputSchema,
                                        },
                                    }
                                )
                            print(
                                f"[MCP] {name}: {len(result.tools)} ツール",
                                file=sys.stderr,
                                flush=True,
                            )
                except Exception as e:
                    print(
                        f"[MCP] {name}: ツール一覧取得エラー: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
            return tools

        if not self._server_configs:
            self._openai_tools = []
            return
        self._openai_tools = asyncio.run(_list_all())

    def get_openai_tools(self) -> list[dict]:
        return self._openai_tools

    def call_tool(self, tool_call_name: str, arguments: dict) -> str:
        if "__" not in tool_call_name:
            raise MCPError(
                f"ツール名形式が不正: {tool_call_name}（server__tool 形式が必要）"
            )
        server_name, tool_name = tool_call_name.split("__", 1)
        srv = self._server_configs.get(server_name)
        if not srv:
            raise MCPError(f"不明なサーバ: {server_name}")

        async def _call():
            params = StdioServerParameters(
                command=srv["command"],
                args=srv.get("args", []),
                env={**os.environ, **srv.get("env", {})},
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    texts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            texts.append(item.text)
                        else:
                            texts.append(str(item))
                    return "\n".join(texts)

        return asyncio.run(_call())

    def close(self):
        self._server_configs.clear()
        self._openai_tools = []
        self._loaded = False


mcp_manager = MCPClientManager()
