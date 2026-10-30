@echo off
rem LocalRouter's executor over MCP stdio (see the app's README), Windows: dist\mcp.cjs on the app's bundled Node.
setlocal
set ELECTRON_RUN_AS_NODE=1
"%~dp0..\LocalRouter.exe" "%~dp0app\mcp.cjs" %*
