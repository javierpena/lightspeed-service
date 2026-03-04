"""Unit tests for MCP utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools.structured import StructuredTool

from ols import constants
from ols.app.models.config import MCPServerConfig
from ols.utils.mcp_utils import (
    MCPPromptInfo,
    _build_prompt_text,
    build_mcp_config,
    fetch_mcp_prompt,
    gather_mcp_prompts,
    gather_mcp_tools,
    get_mcp_prompts,
    get_mcp_tools,
    get_servers_requiring_client_headers,
    match_mcp_prompt,
    resolve_header_value,
    resolve_server_headers,
)


@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP server config."""
    server = MagicMock(spec=MCPServerConfig)
    server.name = "test-server"
    server.url = "http://test:8080/mcp"
    server.timeout = None
    server.headers = {}
    server.resolved_headers = {}
    return server


@pytest.fixture
def mock_k8s_server():
    """Create a mock MCP server with k8s authentication."""
    server = MagicMock(spec=MCPServerConfig)
    server.name = "k8s-server"
    server.url = "http://k8s:8080/mcp"
    server.timeout = 30
    server.headers = {"Authorization": constants.MCP_KUBERNETES_PLACEHOLDER}
    server.resolved_headers = {"Authorization": constants.MCP_KUBERNETES_PLACEHOLDER}
    return server


@pytest.fixture
def mock_client_server():
    """Create a mock MCP server with client authentication."""
    server = MagicMock(spec=MCPServerConfig)
    server.name = "client-server"
    server.url = "http://client:8080/mcp"
    server.timeout = None
    server.headers = {"Authorization": constants.MCP_CLIENT_PLACEHOLDER}
    server.resolved_headers = {"Authorization": constants.MCP_CLIENT_PLACEHOLDER}
    return server


@pytest.fixture
def mock_file_server():
    """Create a mock MCP server with file-based authentication."""
    server = MagicMock(spec=MCPServerConfig)
    server.name = "file-server"
    server.url = "http://file:8080/mcp"
    server.timeout = None
    server.headers = {"Authorization": "Bearer file-token"}
    server.resolved_headers = {"Authorization": "Bearer file-token"}
    return server


@pytest.fixture
def mock_tool():
    """Create a mock StructuredTool."""
    tool = MagicMock(spec=StructuredTool)
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.metadata = {}
    tool.args_schema = {"type": "object", "properties": {}}
    return tool


class TestGetServersRequiringClientHeaders:
    """Tests for get_servers_requiring_client_headers function."""

    def test_empty_servers(self):
        """Test with no servers configured."""
        result = get_servers_requiring_client_headers(None)
        assert result == {}

    def test_no_client_auth_servers(self, mock_k8s_server, mock_file_server):
        """Test with servers that don't require client headers."""
        mock_servers = MagicMock()
        mock_servers.servers = [mock_k8s_server, mock_file_server]

        result = get_servers_requiring_client_headers(mock_servers)
        assert result == {}

    def test_single_client_auth_server(self, mock_client_server):
        """Test with one server requiring client headers."""
        mock_servers = MagicMock()
        mock_servers.servers = [mock_client_server]

        result = get_servers_requiring_client_headers(mock_servers)
        assert result == {"client-server": ["Authorization"]}

    def test_mixed_servers(self, mock_k8s_server, mock_client_server, mock_file_server):
        """Test with mix of server types."""
        mock_servers = MagicMock()
        mock_servers.servers = [mock_k8s_server, mock_client_server, mock_file_server]

        result = get_servers_requiring_client_headers(mock_servers)
        assert result == {"client-server": ["Authorization"]}

    def test_multiple_client_headers(self):
        """Test server requiring multiple client headers."""
        server = MagicMock(spec=MCPServerConfig)
        server.name = "multi-header-server"
        server.headers = {
            "Authorization": constants.MCP_CLIENT_PLACEHOLDER,
            "X-Custom": constants.MCP_CLIENT_PLACEHOLDER,
        }
        server.resolved_headers = {
            "Authorization": constants.MCP_CLIENT_PLACEHOLDER,
            "X-Custom": constants.MCP_CLIENT_PLACEHOLDER,
        }

        mock_servers = MagicMock()
        mock_servers.servers = [server]

        result = get_servers_requiring_client_headers(mock_servers)
        assert result == {"multi-header-server": ["Authorization", "X-Custom"]}


class TestResolveHeaderValue:
    """Tests for resolve_header_value function."""

    def test_kubernetes_placeholder_with_token(self):
        """Test k8s placeholder resolution with token."""
        result = resolve_header_value(
            constants.MCP_KUBERNETES_PLACEHOLDER,
            "Authorization",
            "test-server",
            "k8s-token-123",
            None,
        )
        assert result == "Bearer k8s-token-123"

    def test_kubernetes_placeholder_without_token(self):
        """Test k8s placeholder resolution without token."""
        result = resolve_header_value(
            constants.MCP_KUBERNETES_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            None,
        )
        assert result is None

    def test_client_placeholder_with_headers(self):
        """Test client placeholder resolution with headers."""
        client_headers = {"test-server": {"Authorization": "Bearer client-token"}}
        result = resolve_header_value(
            constants.MCP_CLIENT_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            client_headers,
        )
        assert result == "Bearer client-token"

    def test_client_placeholder_without_headers(self):
        """Test client placeholder resolution without headers."""
        result = resolve_header_value(
            constants.MCP_CLIENT_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            None,
        )
        assert result is None

    def test_client_placeholder_missing_server(self):
        """Test client placeholder with wrong server name."""
        client_headers = {"other-server": {"Authorization": "Bearer token"}}
        result = resolve_header_value(
            constants.MCP_CLIENT_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            client_headers,
        )
        assert result is None

    def test_client_placeholder_missing_header(self):
        """Test client placeholder with missing header name."""
        client_headers = {"test-server": {"X-Custom": "value"}}
        result = resolve_header_value(
            constants.MCP_CLIENT_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            client_headers,
        )
        assert result is None

    def test_already_resolved_value(self):
        """Test already resolved header value."""
        result = resolve_header_value(
            "Bearer file-token", "Authorization", "test-server", None, None
        )
        assert result == "Bearer file-token"


class TestResolveServerHeaders:
    """Tests for resolve_server_headers function."""

    def test_k8s_auth_with_token(self, mock_k8s_server):
        """Test k8s server with token."""
        result = resolve_server_headers(mock_k8s_server, "k8s-token", None)
        assert result == {"Authorization": "Bearer k8s-token"}

    def test_k8s_auth_without_token(self, mock_k8s_server):
        """Test k8s server without token."""
        result = resolve_server_headers(mock_k8s_server, None, None)
        assert result is None

    def test_client_auth_with_headers(self, mock_client_server):
        """Test client server with headers."""
        client_headers = {"client-server": {"Authorization": "Bearer client-token"}}
        result = resolve_server_headers(mock_client_server, None, client_headers)
        assert result == {"Authorization": "Bearer client-token"}

    def test_client_auth_without_headers(self, mock_client_server):
        """Test client server without headers."""
        result = resolve_server_headers(mock_client_server, None, None)
        assert result is None

    def test_file_auth(self, mock_file_server):
        """Test file-based auth server."""
        result = resolve_server_headers(mock_file_server, None, None)
        assert result == {"Authorization": "Bearer file-token"}

    def test_no_headers(self, mock_mcp_server):
        """Test server without headers."""
        result = resolve_server_headers(mock_mcp_server, None, None)
        assert result == {}


class TestBuildMcpConfig:
    """Tests for build_mcp_config function."""

    def test_empty_servers_list(self):
        """Test with empty servers list."""
        result = build_mcp_config([], None, None)
        assert result == {}

    def test_single_k8s_server(self, mock_k8s_server):
        """Test with single k8s server."""
        result = build_mcp_config([mock_k8s_server], "k8s-token", None)
        assert "k8s-server" in result
        assert result["k8s-server"]["transport"] == "streamable_http"
        assert result["k8s-server"]["url"] == "http://k8s:8080/mcp"
        assert result["k8s-server"]["headers"] == {"Authorization": "Bearer k8s-token"}
        assert result["k8s-server"]["timeout"] == 30

    def test_single_client_server(self, mock_client_server):
        """Test with single client server."""
        client_headers = {"client-server": {"Authorization": "Bearer client-token"}}
        result = build_mcp_config([mock_client_server], None, client_headers)
        assert "client-server" in result
        assert result["client-server"]["headers"] == {
            "Authorization": "Bearer client-token"
        }

    def test_single_file_server(self, mock_file_server):
        """Test with single file server."""
        result = build_mcp_config([mock_file_server], None, None)
        assert "file-server" in result
        assert result["file-server"]["headers"] == {
            "Authorization": "Bearer file-token"
        }

    def test_mixed_servers(self, mock_k8s_server, mock_client_server, mock_file_server):
        """Test with multiple server types."""
        client_headers = {"client-server": {"Authorization": "Bearer client-token"}}
        result = build_mcp_config(
            [mock_k8s_server, mock_client_server, mock_file_server],
            "k8s-token",
            client_headers,
        )
        assert len(result) == 3
        assert "k8s-server" in result
        assert "client-server" in result
        assert "file-server" in result

    def test_skips_unresolvable_server(self, mock_k8s_server, mock_file_server):
        """Test that unresolvable servers are skipped."""
        # k8s server without token should be skipped
        result = build_mcp_config([mock_k8s_server, mock_file_server], None, None)
        assert len(result) == 1
        assert "file-server" in result
        assert "k8s-server" not in result

    def test_server_without_timeout(self, mock_client_server):
        """Test server config without timeout."""
        client_headers = {"client-server": {"Authorization": "Bearer token"}}
        result = build_mcp_config([mock_client_server], None, client_headers)
        assert "timeout" not in result["client-server"]

    def test_error_handling(self):
        """Test error handling in config building."""
        # Create a server that will cause an error
        bad_server = MagicMock(spec=MCPServerConfig)
        bad_server.name = "bad-server"
        bad_server.resolved_headers.items.side_effect = Exception("Test error")

        result = build_mcp_config([bad_server], None, None)
        assert result == {}


@pytest.mark.asyncio
class TestGatherMcpTools:
    """Tests for gather_mcp_tools function."""

    async def test_empty_servers(self):
        """Test with no servers."""
        result = await gather_mcp_tools({})
        assert result == []

    async def test_single_server_success(self, mock_tool):
        """Test gathering tools from one server."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [mock_tool]
            mock_client_cls.return_value = mock_client

            servers = {
                "test-server": {"transport": "streamable_http", "url": "http://test"}
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 1
            assert result[0].name == "test_tool"
            assert result[0].metadata["mcp_server"] == "test-server"

    async def test_multiple_servers(self, mock_tool):
        """Test gathering tools from multiple servers."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool1 = MagicMock(spec=StructuredTool)
            tool1.name = "tool1"
            tool1.metadata = {}
            tool1.args_schema = {"type": "object", "properties": {}}

            tool2 = MagicMock(spec=StructuredTool)
            tool2.name = "tool2"
            tool2.metadata = {}
            tool2.args_schema = {"type": "object", "properties": {}}

            mock_client = AsyncMock()
            mock_client.get_tools.side_effect = [[tool1], [tool2]]
            mock_client_cls.return_value = mock_client

            servers = {
                "server1": {"transport": "streamable_http", "url": "http://s1"},
                "server2": {"transport": "streamable_http", "url": "http://s2"},
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 2
            assert result[0].metadata["mcp_server"] == "server1"
            assert result[1].metadata["mcp_server"] == "server2"

    async def test_server_failure_isolation(self):
        """Test that one server failure doesn't affect others."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool = MagicMock(spec=StructuredTool)
            tool.name = "good_tool"
            tool.metadata = {}
            tool.args_schema = {"type": "object", "properties": {}}

            mock_client = AsyncMock()
            mock_client.get_tools.side_effect = [
                Exception("Server 1 failed"),
                [tool],
            ]
            mock_client_cls.return_value = mock_client

            servers = {
                "bad-server": {"transport": "streamable_http", "url": "http://bad"},
                "good-server": {"transport": "streamable_http", "url": "http://good"},
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 1
            assert result[0].name == "good_tool"
            assert result[0].metadata["mcp_server"] == "good-server"

    async def test_tool_filtering_with_allowlist(self):
        """Test filtering tools by allowed names."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool1 = MagicMock(spec=StructuredTool)
            tool1.name = "allowed_tool"
            tool1.metadata = {}
            tool1.args_schema = {"type": "object", "properties": {}}

            tool2 = MagicMock(spec=StructuredTool)
            tool2.name = "blocked_tool"
            tool2.metadata = {}
            tool2.args_schema = {"type": "object", "properties": {}}

            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [tool1, tool2]
            mock_client_cls.return_value = mock_client

            servers = {
                "test-server": {"transport": "streamable_http", "url": "http://test"}
            }
            result = await gather_mcp_tools(
                servers, allowed_tool_names={"allowed_tool"}
            )

            assert len(result) == 1
            assert result[0].name == "allowed_tool"

    async def test_tool_without_metadata(self):
        """Test handling tools without metadata attribute."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool = MagicMock(spec=StructuredTool)
            tool.name = "test_tool"
            tool.args_schema = {"type": "object", "properties": {}}
            del tool.metadata

            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [tool]
            mock_client_cls.return_value = mock_client

            servers = {
                "test-server": {"transport": "streamable_http", "url": "http://test"}
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 1
            assert hasattr(result[0], "metadata")
            assert result[0].metadata["mcp_server"] == "test-server"

    async def test_tool_with_none_metadata(self):
        """Test handling tools with None metadata."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool = MagicMock(spec=StructuredTool)
            tool.name = "test_tool"
            tool.metadata = None
            tool.args_schema = {"type": "object", "properties": {}}

            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [tool]
            mock_client_cls.return_value = mock_client

            servers = {
                "test-server": {"transport": "streamable_http", "url": "http://test"}
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 1
            assert result[0].metadata is not None
            assert result[0].metadata["mcp_server"] == "test-server"


@pytest.mark.asyncio
class TestGetMcpTools:
    """Tests for get_mcp_tools function."""

    async def test_without_tools_rag(self, mock_file_server, mock_tool):
        """Test getting tools when tools_rag not configured."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            mock_config.tools_rag = None
            mock_config.mcp_servers.servers = [mock_file_server]
            mock_gather.return_value = [mock_tool]

            result = await get_mcp_tools("test query")

            assert len(result) == 1
            assert result[0].name == "test_tool"

    async def test_with_tools_rag_first_call(self, mock_k8s_server, mock_tool):
        """Test first call with tools_rag (cold start)."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            # Setup config
            mock_config.tools_rag = MagicMock()
            mock_config.tools_rag.populate_tools = MagicMock()
            mock_config.tools_rag.set_default_servers = MagicMock()
            mock_config.tools_rag.retrieve_hybrid.return_value = {
                "k8s-server": [{"name": "test_tool", "description": "test"}]
            }
            mock_config.k8s_tools_resolved = False
            mock_config.mcp_servers.servers = [mock_k8s_server]
            mock_config.mcp_servers_dict = {"k8s-server": mock_k8s_server}

            mock_gather.return_value = [mock_tool]

            result = await get_mcp_tools(
                "test query", user_token="k8s-token"  # noqa: S106
            )

            # Should populate tools_rag on first call
            assert mock_config.tools_rag.populate_tools.called
            assert len(result) == 1

    async def test_with_client_headers(
        self, mock_k8s_server, mock_client_server, mock_tool
    ):
        """Test with client headers provided."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            # Setup config
            mock_config.tools_rag = MagicMock()
            mock_config.tools_rag.populate_tools = MagicMock()
            mock_config.tools_rag.set_default_servers = MagicMock()
            mock_config.tools_rag.retrieve_hybrid.return_value = {
                "client-server": [{"name": "test_tool", "description": "test"}]
            }
            mock_config.k8s_tools_resolved = True
            mock_config.mcp_servers.servers = [mock_k8s_server, mock_client_server]
            mock_config.mcp_servers_dict = {
                "k8s-server": mock_k8s_server,
                "client-server": mock_client_server,
            }

            mock_gather.return_value = [mock_tool]

            client_headers = {"client-server": {"Authorization": "Bearer token"}}
            result = await get_mcp_tools(
                "test query",
                user_token="k8s-token",  # noqa: S106
                client_headers=client_headers,
            )

            # Should call retrieve_hybrid with client server names
            mock_config.tools_rag.retrieve_hybrid.assert_called_once()
            call_args = mock_config.tools_rag.retrieve_hybrid.call_args
            assert call_args[0][0] == "test query"  # query argument
            assert call_args[1]["client_servers"] == [
                "client-server"
            ]  # client_servers kwarg
            assert len(result) == 1

    async def test_rag_filtering_error_fallback(self, mock_file_server, mock_tool):
        """Test fallback to all tools when RAG filtering fails."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            # Setup config
            mock_config.tools_rag = MagicMock()
            mock_config.tools_rag.retrieve_hybrid.side_effect = Exception("RAG error")
            mock_config.k8s_tools_resolved = True
            mock_config.mcp_servers.servers = [mock_file_server]
            mock_config.mcp_servers_dict = {"file-server": mock_file_server}

            mock_gather.return_value = [mock_tool]

            result = await get_mcp_tools("test query")

            # Should fallback to all tools
            assert len(result) == 1
            assert result[0].name == "test_tool"

    async def test_no_matching_tools(self, mock_k8s_server):
        """Test when RAG filtering returns no matches."""
        with patch("ols.utils.mcp_utils.config") as mock_config:
            # Setup config
            mock_config.tools_rag = MagicMock()
            mock_config.tools_rag.retrieve_hybrid.return_value = {}
            mock_config.k8s_tools_resolved = True
            mock_config.mcp_servers.servers = [mock_k8s_server]
            mock_config.mcp_servers_dict = {"k8s-server": mock_k8s_server}

            result = await get_mcp_tools(
                "test query", user_token="k8s-token"  # noqa: S106
            )

            # Should return empty list
            assert result == []

    async def test_no_servers_configured(self):
        """Test when no MCP servers are configured."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            mock_config.tools_rag = None
            mock_config.mcp_servers.servers = []
            mock_gather.return_value = []

            result = await get_mcp_tools("test query")

            assert result == []


def _make_prompt_info(
    name: str = "test-prompt",
    description: str = "A test prompt",
    server_name: str = "test-server",
    arguments: list | None = None,
) -> MCPPromptInfo:
    """Create an MCPPromptInfo for tests."""
    return MCPPromptInfo(
        name=name,
        description=description,
        arguments=arguments or [],
        server_name=server_name,
    )


class TestBuildPromptText:
    """Tests for _build_prompt_text helper."""

    def test_name_and_description(self):
        """Test combining name and description."""
        prompt = _make_prompt_info(name="deploy-app", description="Deploy an application")
        assert _build_prompt_text(prompt) == "deploy-app Deploy an application"

    def test_empty_description(self):
        """Test with empty description."""
        prompt = _make_prompt_info(name="run", description="")
        assert _build_prompt_text(prompt) == "run "


class TestMatchMcpPrompt:
    """Tests for match_mcp_prompt function."""

    def test_empty_prompts_list(self):
        """Test with no prompts available."""
        assert match_mcp_prompt("some query", []) is None

    def test_empty_query(self):
        """Test with empty query."""
        prompt = _make_prompt_info()
        assert match_mcp_prompt("", [prompt]) is None

    def test_whitespace_only_query(self):
        """Test with whitespace-only query."""
        prompt = _make_prompt_info()
        assert match_mcp_prompt("   ", [prompt]) is None

    def test_exact_name_match(self):
        """Test that a query matching the prompt name scores highly."""
        prompt = _make_prompt_info(
            name="explain-code", description="Explain source code in detail"
        )
        result = match_mcp_prompt("explain-code", [prompt], threshold=0.1)
        assert result is not None
        assert result["name"] == "explain-code"

    def test_description_match(self):
        """Test matching based on description words."""
        prompt = _make_prompt_info(
            name="review", description="Review kubernetes deployment configuration"
        )
        result = match_mcp_prompt("review kubernetes deployment", [prompt], threshold=0.1)
        assert result is not None
        assert result["name"] == "review"

    def test_best_match_among_multiple(self):
        """Test selecting the best match from multiple prompts."""
        prompts = [
            _make_prompt_info(
                name="deploy", description="Deploy application to cluster"
            ),
            _make_prompt_info(
                name="debug-pods", description="Debug failing pods in namespace"
            ),
        ]
        result = match_mcp_prompt("debug failing pods", prompts, threshold=0.1)
        assert result is not None
        assert result["name"] == "debug-pods"

    def test_below_threshold(self):
        """Test that low-scoring matches are rejected."""
        prompt = _make_prompt_info(
            name="deploy-app", description="Deploy an application to the cluster"
        )
        result = match_mcp_prompt("unrelated weather forecast", [prompt], threshold=0.9)
        assert result is None

    def test_single_prompt_with_overlap(self):
        """Test that a single prompt with word overlap is matched."""
        prompt = _make_prompt_info(name="hello", description="say hello")
        result = match_mcp_prompt("hello", [prompt], threshold=0.1)
        assert result is not None

    def test_no_word_overlap(self):
        """Test that completely unrelated query returns None."""
        prompt = _make_prompt_info(name="deploy", description="deploy application")
        result = match_mcp_prompt("xyz abc", [prompt], threshold=0.01)
        assert result is None


@pytest.mark.asyncio
class TestGatherMcpPrompts:
    """Tests for gather_mcp_prompts function."""

    async def test_empty_servers(self):
        """Test with no servers."""
        result = await gather_mcp_prompts({})
        assert result == []

    async def test_single_server_with_prompts(self):
        """Test gathering prompts from one server."""
        mock_prompt = MagicMock()
        mock_prompt.name = "deploy"
        mock_prompt.description = "Deploy an app"
        mock_prompt.arguments = None

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.prompts = [mock_prompt]
        mock_session.list_prompts.return_value = mock_list_result

        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_client.session.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            servers = {"s1": {"transport": "streamable_http", "url": "http://s1"}}
            result = await gather_mcp_prompts(servers)

        assert len(result) == 1
        assert result[0]["name"] == "deploy"
        assert result[0]["server_name"] == "s1"
        assert result[0]["arguments"] == []

    async def test_prompt_with_arguments(self):
        """Test prompt metadata includes argument details."""
        mock_arg = MagicMock()
        mock_arg.name = "filename"
        mock_arg.description = "Path to file"
        mock_arg.required = True

        mock_prompt = MagicMock()
        mock_prompt.name = "review"
        mock_prompt.description = "Code review"
        mock_prompt.arguments = [mock_arg]

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_list_result.prompts = [mock_prompt]
        mock_session.list_prompts.return_value = mock_list_result

        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_client.session.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            servers = {"s1": {"transport": "streamable_http", "url": "http://s1"}}
            result = await gather_mcp_prompts(servers)

        assert len(result) == 1
        assert len(result[0]["arguments"]) == 1
        assert result[0]["arguments"][0]["name"] == "filename"
        assert result[0]["arguments"][0]["description"] == "Path to file"
        assert result[0]["arguments"][0]["required"] is True

    async def test_server_failure_isolation(self):
        """Test that one server failure does not affect others."""
        mock_prompt = MagicMock()
        mock_prompt.name = "good"
        mock_prompt.description = "Good prompt"
        mock_prompt.arguments = None

        good_session = AsyncMock()
        good_list = MagicMock()
        good_list.prompts = [mock_prompt]
        good_session.list_prompts.return_value = good_list

        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_cls:
            mock_client = MagicMock()

            call_count = 0

            def _session_side_effect(name: str) -> MagicMock:
                nonlocal call_count
                ctx = MagicMock()
                if name == "bad":
                    ctx.__aenter__ = AsyncMock(side_effect=Exception("down"))
                else:
                    ctx.__aenter__ = AsyncMock(return_value=good_session)
                ctx.__aexit__ = AsyncMock(return_value=False)
                call_count += 1
                return ctx

            mock_client.session.side_effect = _session_side_effect
            mock_cls.return_value = mock_client

            servers = {
                "bad": {"transport": "streamable_http", "url": "http://bad"},
                "good": {"transport": "streamable_http", "url": "http://good"},
            }
            result = await gather_mcp_prompts(servers)

        assert len(result) == 1
        assert result[0]["name"] == "good"
        assert result[0]["server_name"] == "good"


@pytest.mark.asyncio
class TestGetMcpPrompts:
    """Tests for get_mcp_prompts function."""

    async def test_no_servers_configured(self):
        """Test returns empty when no MCP servers configured."""
        with patch("ols.utils.mcp_utils.config") as mock_config:
            mock_config.mcp_servers = None
            result = await get_mcp_prompts()
            assert result == []

    async def test_returns_cached_prompts(self):
        """Test that cached prompts are returned on subsequent calls."""
        cached = [_make_prompt_info(name="cached")]
        with patch("ols.utils.mcp_utils.config") as mock_config:
            mock_config.mcp_servers.servers = [MagicMock()]
            mock_config.mcp_prompts_loaded = True
            mock_config._mcp_prompts = cached

            result = await get_mcp_prompts()
            assert result is cached

    async def test_loads_and_caches_prompts(self):
        """Test prompts are loaded from servers and cached."""
        prompt_info = _make_prompt_info(name="fresh")
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.build_mcp_config") as mock_build,
            patch("ols.utils.mcp_utils.gather_mcp_prompts") as mock_gather,
        ):
            mock_config.mcp_servers.servers = [MagicMock()]
            mock_config.mcp_prompts_loaded = False
            mock_config._mcp_prompts = []
            mock_build.return_value = {"s1": {"transport": "streamable_http", "url": "http://s1"}}
            mock_gather.return_value = [prompt_info]

            result = await get_mcp_prompts(user_token="tok")

            assert result == [prompt_info]
            assert mock_config.mcp_prompts_loaded is True
            assert mock_config._mcp_prompts == [prompt_info]

    async def test_empty_servers_config(self):
        """Test returns empty when build_mcp_config yields nothing."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.build_mcp_config") as mock_build,
        ):
            mock_config.mcp_servers.servers = [MagicMock()]
            mock_config.mcp_prompts_loaded = False
            mock_build.return_value = {}

            result = await get_mcp_prompts()
            assert result == []


@pytest.mark.asyncio
class TestFetchMcpPrompt:
    """Tests for fetch_mcp_prompt function."""

    async def test_successful_fetch(self):
        """Test fetching prompt messages."""
        prompt = _make_prompt_info(name="greet", server_name="s1")
        servers_config = {"s1": {"transport": "streamable_http", "url": "http://s1"}}
        expected = [HumanMessage(content="Hello!")]

        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_prompt.return_value = expected
            mock_cls.return_value = mock_client

            result = await fetch_mcp_prompt(prompt, servers_config)

        assert result == expected
        mock_client.get_prompt.assert_called_once_with("s1", "greet", arguments=None)

    async def test_fetch_with_arguments(self):
        """Test fetching prompt with arguments."""
        prompt = _make_prompt_info(name="review", server_name="s1")
        servers_config = {"s1": {"transport": "streamable_http", "url": "http://s1"}}
        args = {"filename": "main.py"}

        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_prompt.return_value = [HumanMessage(content="Review main.py")]
            mock_cls.return_value = mock_client

            result = await fetch_mcp_prompt(prompt, servers_config, arguments=args)

        assert len(result) == 1
        mock_client.get_prompt.assert_called_once_with(
            "s1", "review", arguments={"filename": "main.py"}
        )

    async def test_server_not_in_config(self):
        """Test returns empty when server is missing from config."""
        prompt = _make_prompt_info(name="missing", server_name="absent")
        result = await fetch_mcp_prompt(prompt, {})
        assert result == []

    async def test_fetch_error(self):
        """Test returns empty on server error."""
        prompt = _make_prompt_info(name="fail", server_name="s1")
        servers_config = {"s1": {"transport": "streamable_http", "url": "http://s1"}}

        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_prompt.side_effect = Exception("connection refused")
            mock_cls.return_value = mock_client

            result = await fetch_mcp_prompt(prompt, servers_config)

        assert result == []

    async def test_multi_message_prompt(self):
        """Test prompt returning multiple messages."""
        prompt = _make_prompt_info(name="chat", server_name="s1")
        servers_config = {"s1": {"transport": "streamable_http", "url": "http://s1"}}
        messages = [
            HumanMessage(content="Explain this code"),
            AIMessage(content="I'll analyze it step by step."),
            HumanMessage(content="Focus on error handling"),
        ]

        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_prompt.return_value = messages
            mock_cls.return_value = mock_client

            result = await fetch_mcp_prompt(prompt, servers_config)

        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], HumanMessage)


