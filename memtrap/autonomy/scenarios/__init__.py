"""Turn scenario records (data/autonomy/*.json) into runnable scenarios.

A runnable scenario is a dict with `task` (the user's request), `tools` ({name: "name(args): description"}),
`sim(name, args, state)` (the simulated tool result), `destructive` (the irreversible action) and `det(trajectory)`.
"""
