# Unit tests for the assistant's highest-risk logic: the parts where a bug produces
# a confident answer with no evidence behind it, and nothing visibly fails.
#
# Deliberately not tested here (lower stakes, or fails loudly on the first real call):
# eval and token telemetry, previous_response_id handling in run_assistant, tool
# schemas and adapters, and the prompt text (model behavior, measured by the eval).
# Balancer's search tool is tested in examples/balancer/test_search_tool.py.

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from tiny_agent.agentic_loop import handle_tool_calls, run_agentic_loop
from tiny_agent.assistant_types import Tool, ToolCallStatus


# Minimal stand-ins for OpenAI response objects: only the attributes the loop reads.
def call(name, call_id, **arguments):
    return SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments=json.dumps(arguments))


def response(response_id, *items, text=None):
    return SimpleNamespace(id=response_id, output=list(items), output_text=text)


# --- The loop grounds the answer in tool output --------------------------------------
#
# If a tool's output is dropped, repeated, or paired with the wrong call_id, the model
# still writes a fluent answer, just without the retrieved evidence. The loop must also
# stop once the model stops calling tools; a reasoning item is not a tool call.

def test_loop_feeds_each_tool_output_back_until_the_model_answers():
    tool = Tool("search_documents", "", {}, run=MagicMock(side_effect=["out1", "out2"]))
    context = object()
    client = MagicMock()
    # A list side_effect runs out: if the loop failed to stop, the extra call would raise.
    client.responses.create.side_effect = [
        response("r2", call("search_documents", "c2", query="q2")),
        response("r3", SimpleNamespace(type="reasoning"), text="answer"),
    ]

    result = run_agentic_loop(
        response("r1", call("search_documents", "c1", query="q1")), client, {"instructions": "cite sources"}, [tool], context
    )

    # Each follow-up carries the previous iteration's output, paired with its call_id,
    # and chains off that iteration's response id. It also resends the model settings:
    # instructions are not carried over through previous_response_id, so without them
    # the final answer would be written without the system prompt.
    assert [
        (c.kwargs["previous_response_id"], c.kwargs["instructions"], c.kwargs["input"])
        for c in client.responses.create.call_args_list
    ] == [
        ("r1", "cite sources", [{"type": "function_call_output", "call_id": "c1", "output": "out1"}]),
        ("r2", "cite sources", [{"type": "function_call_output", "call_id": "c2", "output": "out2"}]),
    ]
    # Every dispatch gets the run's context, unchanged.
    assert [c.kwargs["context"] for c in tool.run.call_args_list] == [context, context]
    assert (result.output_text, result.response_id) == ("answer", "r3")


# --- Tool failures are reported, never hidden ----------------------------------------
#
# A failed call must reach the model as an error message (so it can retry or say it
# found nothing) and be recorded with its own status. A failure that raises out of the
# loop fails the user's request; one that looks like success turns into a confident,
# unsupported answer.

def test_tool_failures_reach_the_model_and_are_recorded_by_kind():
    broken = Tool("search_documents", "", {}, run=MagicMock(side_effect=RuntimeError("db down")))

    outputs, records = handle_tool_calls(
        response("r1", call("made_up_tool", "c1"), call("search_documents", "c2", query="q")), [broken], context=None
    )

    assert [o["call_id"] for o in outputs] == ["c1", "c2"]
    assert "made_up_tool" in outputs[0]["output"] and "db down" in outputs[1]["output"]
    # Opposite diagnoses: the model invented a tool, versus our tool broke.
    assert [r.status for r in records] == [ToolCallStatus.UNREGISTERED, ToolCallStatus.FAILED]
