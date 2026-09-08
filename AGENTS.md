## Agent Task — keep tickets updated

When you work on an Agent Task ticket (e.g. an `AI-…` code) in this repo, keep the ticket a live
record via the Agent Task MCP tools (`add_comment`, `update_task`):

- As you make progress, post short `add_comment` updates at meaningful checkpoints (a sub-goal done,
  a blocker hit/cleared, a decision made) — 1–3 lines: what changed, what's next.
- When you open a PR for the ticket, record it immediately: `update_task({ taskId, prUrl })`, and
  mention it in a comment. Keep `prUrl` current if the PR is replaced.
- When the work is done, leave a final comment summarizing the latest changes and link the merged
  PR. Don't set `status: done` silently — confirm with the user (and that the PR is merged) first.
