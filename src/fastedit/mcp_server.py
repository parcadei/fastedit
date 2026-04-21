"""MCP server facade — re-exports from fastedit.mcp for backward compatibility.

The actual implementation lives in fastedit.mcp.server (core),
fastedit.mcp.tools_edit, fastedit.mcp.tools_read, and fastedit.mcp.tools_ast.

This module exists so that:
- `python -m fastedit.mcp_server` still works
- Existing MCP configs pointing to fastedit.mcp_server still work
"""

from .mcp.server import (  # noqa: F401
    BackupStore,
    ModelPool,
    _atomic_write,
    lifespan,
    main,
    mcp,
)
from .mcp.tools_ast import fast_delete, fast_move, fast_rename, fast_undo  # noqa: F401
from .mcp.tools_edit import fast_batch_edit, fast_edit, fast_multi_edit  # noqa: F401
from .mcp.tools_read import fast_diff, fast_read, fast_search  # noqa: F401

if __name__ == "__main__":
    main()
