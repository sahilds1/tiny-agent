# Balancer example

How Balancer's assistant uses the agentic loop, copied from Balancer's
`server/api/views/assistant/` at the fork point (`3bd48f4`, PR #524).

These files are for reference only. They import Balancer's Django paths (`api.views.assistant.*`,
`api.services.*`), so they don't run in this repo, and they aren't installed or tested.

| File | What it shows |
|---|---|
| `assistant_services.py` | An app building the client and model settings and calling the loop, with the request user as its context |
| `tool_services.py` | Defining tools and the `TOOLS` list |
| `search_tool.py` | A tool that uses the context to scope a search to the request user |
| `assistant_prompts.py` | Where the system prompt lives (blank here) |
| `eval_assistant.py` | An eval runner: concurrent runs, token totals, results CSV |
| `review.ipynb` | Comparing two eval CSVs side by side |
| `test_search_tool.py` | Testing a context-scoped tool through the loop, with the search backend mocked |
