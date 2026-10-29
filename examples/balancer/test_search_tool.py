# Balancer's test of its search tool through the loop, moved out of the library's tests because
# SEARCH_TOOL needs Balancer's embedding search. Reference only: it imports Balancer's Django
# paths, so it doesn't run in this repo.
#
# TODO: Pin search_documents' result format once `File: {file_id}` is dropped (search_tool.py)

import json
from types import SimpleNamespace
from unittest.mock import patch

from api.views.assistant.agentic_loop import handle_tool_calls
from api.views.assistant.assistant_types import ToolCallStatus
from api.views.assistant.tool_services import SEARCH_TOOL


# Minimal stand-ins for OpenAI response objects: only the attributes the loop reads.
def call(name, call_id, **arguments):
    return SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments=json.dumps(arguments))


def response(response_id, *items, text=None):
    return SimpleNamespace(id=response_id, output=list(items), output_text=text)


# --- Retrieval is scoped to the request user, and failure is not "no match" ----------
#
# Runs the real SEARCH_TOOL with only the embedding search mocked. The request user
# must reach the search unchanged: it is what scopes document access, and passing the
# wrong one would silently return someone else's documents. And a search that breaks
# must be recorded as FAILED, not reported to the model as "nothing matched".

@patch("api.views.assistant.search_tool.convert_uuids", side_effect=lambda rows: rows)
@patch("api.views.assistant.search_tool.get_closest_embeddings", side_effect=[RuntimeError("index down"), []])
def test_search_uses_the_request_user_and_tells_failure_from_no_match(mock_search, _):
    user = object()

    outputs, records = handle_tool_calls(
        response("r1", call("search_documents", "c1", query="a"), call("search_documents", "c2", query="b")),
        [SEARCH_TOOL],
        user,
    )

    assert [c.kwargs["user"] for c in mock_search.call_args_list] == [user, user]
    assert [r.status for r in records] == [ToolCallStatus.FAILED, ToolCallStatus.OK]
    assert "No relevant documents found" in outputs[1]["output"]
