import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_data  # noqa: E402


class Select(unittest.TestCase):
    def test_mbpp_sorted_numerically_and_normalized(self):
        rows = [{"task_id": "Mbpp/12", "prompt": "p12", "code": "c", "test_imports": "[]", "test_list": "['assert 1']"},
                {"task_id": "2", "prompt": "p2", "code": "c", "test_imports": [], "test_list": ["assert 2"]},
                {"task_id": "Mbpp/3", "prompt": "p3", "code": "c", "test_imports": None, "test_list": ["assert 3"]}]
        out = fetch_data.select_mbpp(rows, n=2)
        self.assertEqual([r["task_id"] for r in out], ["2", "3"])
        self.assertEqual(out[0]["test_list"], ["assert 2"])
        self.assertEqual(out[1]["test_imports"], [])

    def test_ifeval_keeps_supported_lowest_keys(self):
        rows = [{"key": 30, "prompt": "c", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
                {"key": 10, "prompt": "a", "instruction_id_list": ["language:response_language"], "kwargs": [{}]},
                {"key": 20, "prompt": "b", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}]
        out = fetch_data.select_ifeval(rows, n=5)
        self.assertEqual([r["key"] for r in out], [20, 30])

    def test_ifeval_default_is_the_first_200_supported(self):
        rows = [{"key": k, "prompt": "p", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
                for k in range(300, 0, -1)]
        out = fetch_data.select_ifeval(rows)
        self.assertEqual([r["key"] for r in out], list(range(1, 201)))

    def test_select_first(self):
        rows = [{"text": "a", "other": 1}, {"text": "b"}, {"text": "c"}]
        self.assertEqual(fetch_data.select_first(rows, 2), [{"text": "a"}, {"text": "b"}])


class McpCatalog(unittest.TestCase):
    class Client:
        def __init__(self, response):
            self.response, self.calls = response, []

        def post(self, path, body, timeout=300):
            self.calls.append((path, body["method"]))
            return self.response, 0.1

    def test_catalog_is_the_tools_list_response(self):
        response = {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "read_file"}]}}
        client = self.Client(response)
        self.assertEqual(fetch_data.fetch_mcp_catalog(client), response)
        self.assertEqual(client.calls, [("/mcp", "tools/list")])

    def test_empty_catalog_is_refused(self):
        with self.assertRaises(SystemExit):
            fetch_data.fetch_mcp_catalog(self.Client({"result": {"tools": []}}))


class OnlyIfeval(unittest.TestCase):
    """`--only ifeval` regrows ifeval.jsonl alone: a full fetch would re-snapshot ds4.c (edited on this
    branch) as the haystack and refetch the MCP catalog, and older runs would stop being comparable."""

    def test_rewrites_ifeval_and_its_manifest_entry_only(self):
        rows = [{"key": k, "prompt": "p%d" % k, "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
                for k in (3, 1, 2)]
        blob = "".join(json.dumps(r) + "\n" for r in rows).encode()
        with tempfile.TemporaryDirectory() as d:
            data = pathlib.Path(d)
            (data / "haystack.c").write_bytes(b"old haystack")
            (data / "ifeval.jsonl").write_bytes(b"old ifeval\n")
            other = {"haystack.c": {"bytes": 12, "sha256": "h"}, "mcp_tools.json": {"tools": 11, "sha256": "m"}}
            (data / "manifest.json").write_text(json.dumps(dict(other, **{"ifeval.jsonl": {"rows": 1, "sha256": "x"}})))
            with mock.patch.object(fetch_data, "DATA", data), \
                    mock.patch.object(fetch_data, "_get", return_value=blob) as get, \
                    mock.patch.object(fetch_data, "fetch_mcp_catalog", side_effect=AssertionError("no catalog")), \
                    mock.patch.object(fetch_data, "fetch_rows", side_effect=AssertionError("no other sets")):
                fetch_data.main(["--only", "ifeval"])
            get.assert_called_once_with(fetch_data.IFEVAL_URL)
            written = (data / "ifeval.jsonl").read_bytes()
            self.assertEqual([json.loads(line)["key"] for line in written.splitlines()], [1, 2, 3])
            manifest = json.loads((data / "manifest.json").read_text())
            self.assertEqual(manifest["ifeval.jsonl"],
                             {"rows": 3, "sha256": fetch_data.hashlib.sha256(written).hexdigest()})
            self.assertEqual({k: v for k, v in manifest.items() if k != "ifeval.jsonl"}, other)
            self.assertEqual((data / "haystack.c").read_bytes(), b"old haystack")


class Haystack(unittest.TestCase):
    def test_snapshot_is_ds4_c(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "ds4.c").write_bytes(b"int main(void) { return 0; }\n")
            self.assertEqual(fetch_data.snapshot_haystack(pathlib.Path(d)), b"int main(void) { return 0; }\n")


class ViData(unittest.TestCase):
    def test_knowledge_file(self):
        items = json.loads((HERE / "data" / "vi_knowledge.json").read_text())
        self.assertEqual(len(items), 30)
        self.assertEqual(len({i["id"] for i in items}), 30)
        for item in items:
            self.assertTrue(item["q"].strip())
            self.assertTrue(item["groups"] and all(g and all(k.strip() for k in g) for g in item["groups"]))

    def test_tools_neg_extra_file(self):
        # 24 harness cases on top of the gateway's 6 make tools_neg 30; the negx_ prefix keeps them
        # apart from the gateway's neg_ ids.
        items = json.loads((HERE / "data" / "tools_neg_extra.json").read_text())
        self.assertEqual(len(items), 24)
        self.assertEqual(len({i["id"] for i in items}), 24)
        self.assertEqual(len({i["task"] for i in items}), 24)
        for item in items:
            self.assertEqual(set(item), {"id", "task"})
            self.assertTrue(item["id"].startswith("negx_"))
            self.assertTrue(item["task"].strip())

    def test_writing_file(self):
        items = json.loads((HERE / "data" / "vi_writing.json").read_text())
        self.assertEqual(len(items), 10)
        self.assertEqual(len({i["id"] for i in items}), 10)


if __name__ == "__main__":
    unittest.main()
