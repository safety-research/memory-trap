"""The memory tool, executed in-process.

Formation sessions give the model the API-native memory tool (`memory_20250818`; GPT models get the same commands as
a function) and run each of its calls here, against a {path: text} store. The note is the text of the files the model
wrote.
"""

class FileMemory:
    """Anthropic's ``memory_20250818`` memory tool over an in-memory store.

    A single ``memory`` tool with the six client-side commands — ``view`` / ``create`` /
    ``str_replace`` / ``insert`` / ``delete`` / ``rename`` — over a ``/memories`` directory
    (``dict[path -> text]``). Result strings follow the tool's documentation
    (platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool), and paths outside
    ``/memories`` are rejected. ``FileMemory._run(store, args)`` executes one tool call.
    """

    ROOT = "/memories"
    _ROOTS = frozenset({"", "/", "/memories", "/memories/"})

    @classmethod
    def _run(cls, store: dict, args: dict) -> str:
        # reject paths outside /memories
        for key in ("path", "old_path", "new_path"):
            v = args.get(key)
            if isinstance(v, str) and (".." in v or (v not in cls._ROOTS
                                                     and not v.startswith(cls.ROOT + "/"))):
                return f"Error: path {v} is outside {cls.ROOT}"
        cmd = str(args.get("command") or "")
        if cmd == "view":
            return cls._view(store, args)
        if cmd == "create":
            # `create` overwrites an existing file
            path = str(args.get("path") or "")
            store[path] = str(args.get("file_text") or "")
            return f"File created successfully at: {path}"
        if cmd == "str_replace":
            path = str(args.get("path") or "")
            if path not in store:
                return f"Error: The path {path} does not exist. Please provide a valid path."
            old, new = str(args.get("old_str") or ""), str(args.get("new_str") or "")
            n = store[path].count(old) if old else 0
            if n == 0:
                return (f"No replacement was performed, old_str `{old}` did not appear "
                        f"verbatim in {path}.")
            if n > 1:
                lines = [i + 1 for i, ln in enumerate(store[path].split("\n")) if old in ln]
                return (f"No replacement was performed. Multiple occurrences of old_str "
                        f"`{old}` in lines: {lines}. Please ensure it is unique")
            store[path] = store[path].replace(old, new, 1)
            # success is followed by the edited file with line numbers
            return "The memory file has been edited.\n" + cls._number_lines(store[path])
        if cmd == "insert":
            path = str(args.get("path") or "")
            if path not in store:
                return f"Error: The path {path} does not exist"
            lines = store[path].split("\n")
            try:
                il = int(args.get("insert_line"))
            except (TypeError, ValueError):
                il = -1
            if il < 0 or il > len(lines):
                return (f"Error: Invalid `insert_line` parameter: {il}. It should be within "
                        f"the range of lines of the file: [0, {len(lines)}]")
            lines.insert(il, str(args.get("insert_text") or "").rstrip("\n"))
            store[path] = "\n".join(lines)
            return f"The file {path} has been edited."
        if cmd == "delete":
            path = str(args.get("path") or "")
            if path in cls._ROOTS:
                return f"Error: Cannot delete the memory root {cls.ROOT}"
            if path in store:
                del store[path]
                return f"Successfully deleted {path}"
            subs = [k for k in store if k.startswith(path.rstrip("/") + "/")]
            if subs:
                for k in subs:
                    del store[k]
                return f"Successfully deleted {path}"
            return f"Error: The path {path} does not exist"
        if cmd == "rename":
            old, new = str(args.get("old_path") or ""), str(args.get("new_path") or "")
            if old in cls._ROOTS:
                return f"Error: Cannot rename the memory root {cls.ROOT}"
            if old in store:
                if new in store:
                    return f"Error: The destination {new} already exists"
                store[new] = store.pop(old)
                return f"Successfully renamed {old} to {new}"
            subs = [k for k in store if k.startswith(old.rstrip("/") + "/")]
            if subs:
                for k in subs:
                    store[new + k[len(old):]] = store.pop(k)
                return f"Successfully renamed {old} to {new}"
            return f"Error: The path {old} does not exist"
        return f"Error: unknown command {cmd!r}"

    @classmethod
    def _view(cls, store: dict, args: dict) -> str:
        path = str(args.get("path") or "")
        if path in cls._ROOTS:
            total = sum(len(t) for t in store.values())
            header = (f"Here're the files and directories up to 2 levels deep in {cls.ROOT}, "
                      "excluding hidden items and node_modules:")
            rows = [f"{cls._size(total)}\t{cls.ROOT}"]
            rows += [f"{cls._size(len(store[p]))}\t{p}" for p in sorted(store)]
            return header + "\n" + "\n".join(rows)
        if path not in store:
            return f"The path {path} does not exist. Please provide a valid path."
        text = store[path]
        vr = args.get("view_range")
        # files with more than 999,999 lines return an error, checked before truncation
        if text.count("\n") + 1 > 999999:
            return f"File {path} exceeds maximum line limit of 999,999 lines."
        # a full view shows the first 16,000 characters; a ranged view is not truncated
        truncated = len(text) > 16000 and not (isinstance(vr, (list, tuple)) and len(vr) == 2)
        if truncated:
            text = text[:16000]
        all_lines = text.split("\n")
        if isinstance(vr, (list, tuple)) and len(vr) == 2:
            s, e = int(vr[0]), int(vr[1])
            e = len(all_lines) if e == -1 else e
            sel = [(i + 1, all_lines[i]) for i in range(max(0, s - 1), min(len(all_lines), e))]
            body = "\n".join(f"{i:>6}\t{ln}" for i, ln in sel)
        else:
            body = cls._number_lines(text)
        out = f"Here's the content of {path} with line numbers:\n{body}"
        return out + "\n... (truncated at 16000 characters)" if truncated else out

    @staticmethod
    def _number_lines(text: str) -> str:
        return "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(text.split("\n"), 1))

    @staticmethod
    def _size(n: int) -> str:
        if n >= 1024 * 1024:
            return f"{n / 1024 / 1024:.1f}M"
        return f"{max(0.1, n / 1024):.1f}K"
