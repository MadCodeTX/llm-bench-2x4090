"""
Wrapper for Qwen3CoderToolParser that fixes two upstream bugs:

1. Parser declares self.streamed_args_for_tool but never appends to it,
   causing IndexError in serving.py at the end-of-stream trailer path.

2. Parser stores prev_tool_call_arr[i]['arguments'] as a JSON STRING,
   but serving.py json.dumps() it again to compute the 'expected_call' for
   the trailer — producing a doubly-encoded string that gets concatenated
   onto the streamed args, corrupting the final JSON.

We normalize 'arguments' to a dict so json.dumps yields the same shape the
parser emitted incrementally.
"""
import json

from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager


@ToolParserManager.register_module(["qwen3_coder_fixed"])
class Qwen3CoderToolParserFixed(Qwen3CoderToolParser):
    def extract_tool_calls_streaming(self, *args, **kwargs):
        delta = super().extract_tool_calls_streaming(*args, **kwargs)

        # 1. Maintain streamed_args_for_tool so serving.py can compute the
        #    correct trailing diff.
        if delta and getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = getattr(tc, "index", None)
                if idx is None:
                    continue
                while len(self.streamed_args_for_tool) <= idx:
                    self.streamed_args_for_tool.append("")
                fn = getattr(tc, "function", None)
                args_chunk = getattr(fn, "arguments", None) if fn else None
                if args_chunk:
                    self.streamed_args_for_tool[idx] += args_chunk

        # 2. Normalize prev_tool_call_arr[*]['arguments'] from JSON string -> dict
        #    so serving.py's json.dumps produces the expected unescaped form.
        for entry in self.prev_tool_call_arr:
            a = entry.get("arguments")
            if isinstance(a, str):
                try:
                    entry["arguments"] = json.loads(a)
                except Exception:
                    entry["arguments"] = {}
        return delta
