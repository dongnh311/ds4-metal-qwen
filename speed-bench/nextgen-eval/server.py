"""Run one ds4-server arm, send chat requests, and read its log and the machine's memory (stdlib only)."""
import json
import os
import pathlib
import re
import subprocess
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = pathlib.Path.home() / ".local/ai-gateway/runtime-registry.json"


def registry_command(registry, model=None):
    """The ds4 process_command of registry model `model`, or of the single enabled ds4 runtime."""
    if model:
        return list(registry["models"][model]["runtimes"]["ds4"]["process_command"])
    enabled = [name for name, m in registry["models"].items()
               if m.get("runtimes", {}).get("ds4", {}).get("enabled")]
    if len(enabled) != 1:
        raise SystemExit("expected exactly one enabled ds4 runtime in the registry, found %d (%s); name one "
                         "with \"registry_model\" in the arm config" % (len(enabled), ", ".join(enabled)))
    return list(registry["models"][enabled[0]]["runtimes"]["ds4"]["process_command"])


def split_env(cmd):
    """['/usr/bin/env', 'K=V', ..., binary, args...] -> ({K: V}, [binary, args...])."""
    env, i = {}, 0
    if cmd and os.path.basename(cmd[0]) == "env":
        i = 1
        while i < len(cmd) and "=" in cmd[i] and not cmd[i].startswith("-"):
            key, value = cmd[i].split("=", 1)
            env[key] = value
            i += 1
    return env, list(cmd[i:])


def argv_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def _set_flag(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]


def _remove_flag(argv, flag):
    """Drop a flag and, when the next token is not another flag, its value."""
    while flag in argv:
        i = argv.index(flag)
        end = i + 2 if i + 1 < len(argv) and not argv[i + 1].startswith("-") else i + 1
        del argv[i:end]


_ALIASES = {"--ctx": "-c", "--model": "-m"}


def _add_args(argv, extra):
    """Append `extra`; a flag that argv already has gets its value replaced, never a second copy
    (ds4-server keeps the last copy, argv_value() reads the first)."""
    extra = [_ALIASES.get(a, a) for a in extra]
    i = 0
    while i < len(extra):
        flag = extra[i]
        if i + 1 < len(extra) and not extra[i + 1].startswith("-"):
            _set_flag(argv, flag, extra[i + 1])
            i += 2
            continue
        if flag not in argv:
            argv.append(flag)
        i += 1


def resolve(config, registry, root, port, kv_dir):
    if config.get("base") != "registry":
        raise SystemExit("unsupported arm base %r (only 'registry')" % config.get("base"))
    env, argv = split_env(registry_command(registry, config.get("registry_model")))
    argv = [argv[0]] + [_ALIASES.get(a, a) for a in argv[1:]]
    argv[0] = str(pathlib.Path(root) / "ds4-server")
    _set_flag(argv, "--port", str(port))
    _set_flag(argv, "--kv-disk-dir", str(kv_dir))
    if config.get("model"):
        _set_flag(argv, "-m", config["model"])
    for flag in config.get("args_remove", []):
        _remove_flag(argv, _ALIASES.get(flag, flag))
    _add_args(argv, config.get("args_add", []))
    env.update(config.get("env", {}))
    return env, argv


_PROMPT_DONE = re.compile(r"prompt done ([\d.]+)s")
_PREFILL = re.compile(r"prefill chunk \d+/\d+ \([\d.]+%\) chunk=[\d.]+ t/s avg=([\d.]+) t/s")
_THINK = re.compile(r"thinking closed after (\d+) tokens")
_FINISH = re.compile(r"gen=(\d+) finish=(\w+) ([\d.]+)s")
_DECODE = re.compile(r"decoding chunk=[\d.]+ t/s avg=([\d.]+) t/s")


def parse_request_log(segment):
    """Per-request numbers from the ds4-server log lines of one request."""
    def last(rx, cast=float):
        found = rx.findall(segment)
        return cast(found[-1]) if found else None
    fin = _FINISH.findall(segment)
    return {"prefill_s": last(_PROMPT_DONE), "prefill_tps": last(_PREFILL),
            "think_tokens": last(_THINK, int),
            "gen_tokens": int(fin[-1][0]) if fin else None,
            "finish": fin[-1][1] if fin else None,
            "total_s": float(fin[-1][2]) if fin else None,
            "decode_tps": last(_DECODE)}


class LogCursor:
    def __init__(self, path):
        self.path, self.mark = pathlib.Path(path), 0

    def take(self):
        text = self.path.read_text(errors="replace") if self.path.exists() else ""
        new, self.mark = text[self.mark:], len(text)
        return new


def parse_vm_stat(text):
    def field(name):
        m = re.search(r"%s:\s+(\d+)\." % re.escape(name), text)
        return int(m.group(1)) if m else 0
    page = re.search(r"page size of (\d+) bytes", text)
    return {"page_size": int(page.group(1)) if page else 16384,
            "wired_pages": field("Pages wired down"), "swapouts": field("Swapouts")}


class MemSampler:
    """Samples vm_stat every `interval` seconds; stop() returns the peak wired GiB and the swap-outs."""

    def __init__(self, interval=2.0, read=None):
        self.interval = interval
        self._read = read or (lambda: subprocess.run(["vm_stat"], capture_output=True, text=True).stdout)
        self._stop = threading.Event()
        self._thread = None
        self.first = self.last = None
        self.peak_wired = 0

    def _sample(self):
        s = parse_vm_stat(self._read())
        self.last = s
        if self.first is None:
            self.first = s
        self.peak_wired = max(self.peak_wired, s["wired_pages"] * s["page_size"])

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        return {"peak_wired_gib": round(self.peak_wired / 2**30, 2),
                "swapouts": self.last["swapouts"] - self.first["swapouts"]}


def chat(base_url, messages, max_tokens=16384, extra=None, timeout=7200):
    body = {"model": "ds4", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": False}
    body.update(extra or {})
    req = urllib.request.Request(base_url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    choice = out["choices"][0]
    msg = choice["message"]
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "usage": out.get("usage") or {}, "finish_reason": choice.get("finish_reason"),
            "seconds": round(time.time() - t0, 2)}


class Ds4Server:
    def __init__(self, env, argv, log_path, port):
        self.env, self.argv = dict(env), list(argv)
        self.log_path, self.port = pathlib.Path(log_path), port
        self.base_url = "http://127.0.0.1:%d" % port
        self.proc, self._fh = None, None

    def start(self, timeout=900):
        self._fh = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.argv, cwd=str(ROOT), env={**os.environ, **self.env},
                                     stdout=self._fh, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("ds4-server exited with %s, see %s" % (self.proc.returncode, self.log_path))
            try:
                urllib.request.urlopen(self.base_url + "/v1/models", timeout=2).read()
                return
            except OSError:
                time.sleep(1)
        self.stop(drain=0)
        raise RuntimeError("ds4-server not ready after %ds, see %s" % (timeout, self.log_path))

    def stop(self, term_timeout=120, drain=30):
        """SIGTERM and wait. Never SIGKILL: a Metal process killed with -9 can wedge its GGUF until reboot."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=term_timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "ds4-server pid %d ignored SIGTERM for %ds; not sending SIGKILL (it can wedge the GGUF "
                    "until reboot); stop it by hand" % (self.proc.pid, term_timeout))
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if drain:
            time.sleep(drain)  # let the previous model's wired memory drain before the next load
