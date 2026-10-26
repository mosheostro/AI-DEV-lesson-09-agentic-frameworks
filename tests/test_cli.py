"""CLI acceptance tests — SPEC.md §9 and test rows T21–T27.

Run from the SDD folder:  python -m unittest discover -s tests -v
The real prompts (`ask_human` and the interactive request prompt) read from a scripted
`builtins.input`; the model and MCP tools are the deterministic fakes.
"""
import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from fakes import ROOT, WEBEX_ARGS, WEBEX_REQUEST, FakeServer, make_model, text

import supervisor

TESTS = Path(__file__).resolve().parent
THREAD = re.compile(r"case-\d{14}-[0-9a-f]{4}")
POLICY = "what's the remote work policy?"


class ScriptedInput:
    """Stands in for builtins.input: returns the scripted lines, then raises EOFError."""

    def __init__(self, *lines):
        self.lines, self.prompts = list(lines), []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)

    def requests_asked(self) -> int:
        return sum("Request" in p for p in self.prompts)


def must_not_be_called(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")
    return fail


def no_runtime_resources():
    """Patches every way the CLI could reach .env, Bedrock, MCP or the checkpoint file."""
    return [mock.patch.object(supervisor, "load_dotenv", must_not_be_called("load_dotenv")),
            mock.patch.object(supervisor, "get_model", must_not_be_called("get_model")),
            mock.patch.object(supervisor, "load_tools", must_not_be_called("load_tools"))]


async def run(argv, db, *, server=None, model=None, lines=None, real_runtime=False):
    """Runs cli() with the fakes (unless real_runtime) and the scripted input; returns
    (exit code, output, scripted input)."""
    scripted = ScriptedInput(*(lines or []))
    kwargs = {} if real_runtime else {"tools": (server or FakeServer()).tools(),
                                      "model": model or make_model()}
    out = io.StringIO()
    with mock.patch("builtins.input", scripted), redirect_stdout(out):
        code = await supervisor.cli(argv, db_path=db, **kwargs)
    return code, out.getvalue(), scripted


def threads_in(output: str) -> list[str]:
    return list(dict.fromkeys(THREAD.findall(output)))


class CliCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = Path(self._tmp.name) / "checkpoints.sqlite"

    def tearDown(self):
        self._tmp.cleanup()


# ── T21: approval answers ───────────────────────────────────────────────────────


class ApprovalAnswerTests(unittest.TestCase):
    def ask(self, *lines):
        scripted = ScriptedInput(*lines)
        out = io.StringIO()
        with mock.patch("builtins.input", scripted), redirect_stdout(out):
            decision = supervisor.ask_human({})
        return decision, scripted, out.getvalue()

    def test_t21_approve_words(self):
        for word in ("ok", "OK", " Yes ", "approve", "y", "APPROVE"):
            decision, _, _ = self.ask(word)
            self.assertEqual(decision, {"approved": True}, word)

    def test_t21_reject_words(self):
        for word in ("reject", "No", " n "):
            decision, _, _ = self.ask(word, "budget frozen")
            self.assertEqual(decision, {"approved": False, "reason": "budget frozen"}, word)
        decision, _, _ = self.ask("no", "")
        self.assertEqual(decision, {"approved": False, "reason": "rejected by approver"})

    def test_t21_empty_or_unknown_input_asks_again(self):
        decision, scripted, out = self.ask("", "maybe", "ok")
        self.assertEqual(decision, {"approved": True})
        self.assertEqual(len(scripted.prompts), 3)
        for word in ("approve", "yes", "ok", "reject", "no"):
            self.assertIn(word, out.lower())

    def test_t21_end_of_input_pauses(self):
        with self.assertRaises(EOFError):
            self.ask("", "maybe")


# ── T22: approval end to end through the CLI ────────────────────────────────────


class ApprovalCliTests(CliCase):
    async def test_t22_ok_approves_exactly_the_shown_args(self):
        server = FakeServer()
        code, out, _ = await run([], self.db, server=server, lines=["ok"])
        self.assertEqual(code, 0, out)
        (thread,) = threads_in(out)
        self.assertIn("create_access_request", out)
        self.assertIn("employee_id: E010", out)
        self.assertIn("software: Webex", out)
        self.assertIn("evidence", out.lower())
        self.assertEqual(server.writes, [WEBEX_ARGS])
        self.assertIn("Outcome: access request filed", out)

    async def test_t22_reject_after_invalid_input_never_writes(self):
        server = FakeServer()
        code, out, scripted = await run([], self.db, server=server, lines=["", "nope", "no", "budget frozen"])
        self.assertEqual(code, 0, out)
        self.assertEqual(server.writes, [])
        self.assertIn("Outcome: rejected by the approver", out)
        self.assertEqual(len(scripted.prompts), 4)


# ── T23: help and argument parsing ──────────────────────────────────────────────


class HelpTests(CliCase):
    async def test_t23_help_loads_nothing(self):
        patches = no_runtime_resources() + [mock.patch.object(
            supervisor.AsyncSqliteSaver, "from_conn_string", must_not_be_called("SQLite saver"))]
        for flag in ("--help", "-h"):
            for p in patches:
                p.start()
            try:
                code, out, _ = await run([flag], self.db, real_runtime=True)
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(code, 0, out)
            for mode in ("--demo", "--interactive", "--resume", "--status", "--help"):
                self.assertIn(mode, out)
            self.assertFalse(self.db.exists())

    async def test_t23_bad_options_are_usage_errors(self):
        for p in no_runtime_resources():
            p.start()
        try:
            for argv in (["--bogus"], ["--demo", "x"], ["--resume"], ["--status", "a", "b"],
                         ["-i", "extra"], ["--interactiv"]):
                code, out, _ = await run(argv, self.db, real_runtime=True)
                self.assertEqual(code, 2, argv)
                self.assertIn("Usage", out, argv)
        finally:
            mock.patch.stopall()

    async def test_t23_double_dash_passes_a_request_starting_with_dash(self):
        code, out, _ = await run(["--", "-x", "policy?"], self.db, model=make_model(workers=["knowledge"], may_file=False))
        self.assertEqual(code, 0, out)
        self.assertIn("You: -x policy?", out)

    async def test_t23_a_request_containing_help_is_still_a_request(self):
        code, out, _ = await run(["what does --help do"], self.db,
                                 model=make_model(workers=[], may_file=False))
        self.assertEqual(code, 0, out)
        self.assertIn("You: what does --help do", out)

    def test_t23_help_subprocess_without_environment(self):
        script = textwrap.dedent(f"""
            import asyncio, os, sys
            sys.path.insert(0, {str(ROOT)!r})
            import supervisor
            code = asyncio.run(supervisor.cli(["--help"]))
            print("ENV_LOADED" if "MCP_SERVER_URL" in os.environ else "ENV_UNTOUCHED")
            sys.exit(code)
        """)
        env = {k: v for k, v in os.environ.items() if k not in supervisor.REQUIRED_ENV}
        env["PYTHONIOENCODING"] = "utf-8"
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                              encoding="utf-8", timeout=30, env=env, cwd=tempfile.gettempdir())
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("--interactive", done.stdout)
        self.assertIn("ENV_UNTOUCHED", done.stdout)


# ── T24: demo and one-shot ──────────────────────────────────────────────────────


class DemoAndOneShotTests(CliCase):
    async def test_t24_no_argument_and_demo_run_the_webex_case_once(self):
        for argv in ([], ["--demo"]):
            server = FakeServer()
            code, out, _ = await run(argv, self.db, server=server, lines=["ok"])
            self.assertEqual(code, 0, (argv, out))
            self.assertIn(f"You: {WEBEX_REQUEST}", out)
            self.assertEqual(len(threads_in(out)), 1)
            self.assertEqual(len(server.writes), 1)

    async def test_t24_one_shot_request(self):
        server = FakeServer()
        code, out, scripted = await run([POLICY], self.db, server=server,
                                        model=make_model(workers=["knowledge"], may_file=False))
        self.assertEqual(code, 0, out)
        self.assertIn(f"You: {POLICY}", out)
        self.assertEqual(len(threads_in(out)), 1)
        self.assertEqual(scripted.prompts, [])
        self.assertEqual(server.writes, [])
        self.assertIn("Outcome: no access request was proposed", out)


# ── T25: interactive session ────────────────────────────────────────────────────


def two_plans():
    return make_model(plans=[supervisor.Plan(workers=["records", "eligibility"], may_file_request=True),
                             supervisor.Plan(workers=["knowledge"], may_file_request=False)])


class InteractiveTests(CliCase):
    async def test_t25_several_requests_each_on_a_new_thread(self):
        server = FakeServer()
        code, out, scripted = await run(["--interactive"], self.db, server=server, model=two_plans(),
                                        lines=[WEBEX_REQUEST, "ok", POLICY, "exit"])
        self.assertEqual(code, 0, out)
        self.assertEqual(len(threads_in(out)), 2)
        self.assertEqual(len(server.writes), 1)
        self.assertEqual(out.count("NovaOps:"), 2)
        self.assertEqual(scripted.requests_asked(), 3)          # asked again after each answer

    async def test_t25_quit_end_of_input_and_blank_lines(self):
        for lines in (["quit"], ["QUIT"], [], ["", "  ", "exit"]):
            code, out, scripted = await run(["-i"], self.db, lines=list(lines))
            self.assertEqual(code, 0, (lines, out))
            self.assertEqual(threads_in(out), [], lines)

    async def test_t25_paused_approval_ends_the_session_and_resumes_later(self):
        server = FakeServer()
        code, out, scripted = await run(["-i"], self.db, server=server, model=two_plans(),
                                        lines=[WEBEX_REQUEST])          # end-of-input at the approval
        self.assertEqual(code, 0, out)
        self.assertIn("Paused", out)
        (thread,) = threads_in(out)
        self.assertIn(f"python supervisor.py --resume {thread}", out)
        self.assertEqual(server.writes, [])

        def ctrl_c(payload):
            raise KeyboardInterrupt

        scripted = ScriptedInput(WEBEX_REQUEST, POLICY)              # Ctrl-C at the approval prompt
        out_io = io.StringIO()
        with mock.patch("builtins.input", scripted), redirect_stdout(out_io):
            code = await supervisor.cli(["-i"], tools=server.tools(), model=two_plans(),
                                        db_path=self.db, ask=ctrl_c)
        self.assertEqual(code, 0, out_io.getvalue())
        self.assertIn("Paused", out_io.getvalue())
        self.assertEqual(scripted.lines, [POLICY])                   # the next request is never read

        code, out, _ = await run(["--resume", thread], self.db, server=server, lines=["approve"])
        self.assertEqual(code, 0, out)
        self.assertEqual(len(server.writes), 1)

    async def test_t25_runtime_error_returns_to_the_prompt_without_retrying(self):
        server, act_calls = FakeServer(), []

        def act(messages, bound):
            act_calls.append(bound)
            if len(act_calls) == 1:
                raise RuntimeError("Bedrock unavailable")
            return text("Here is the policy.")

        model = make_model(plans=[supervisor.Plan(workers=["records", "eligibility"], may_file_request=True),
                                  supervisor.Plan(workers=["knowledge"], may_file_request=False)], act=act)
        code, out, scripted = await run(["-i"], self.db, server=server, model=model,
                                        lines=[WEBEX_REQUEST, POLICY, "exit"])
        self.assertEqual(code, 0, out)
        self.assertIn("RuntimeError: Bedrock unavailable", out)
        self.assertIn("nothing was filed", out.lower())
        self.assertEqual(len(threads_in(out)), 2)
        self.assertEqual(len(act_calls), 2)                         # failed request not retried
        self.assertEqual(server.writes, [])
        self.assertEqual(scripted.requests_asked(), 3)

    async def test_t25_uncertain_write_is_reported_and_never_retried(self):
        server = FakeServer(write_mode="transport")
        code, out, scripted = await run(["-i"], self.db, server=server,
                                        lines=[WEBEX_REQUEST, "ok", "exit"])
        self.assertEqual(code, 0, out)
        self.assertIn("uncertain", out.lower())
        self.assertEqual(len(server.writes), 1)
        self.assertEqual(scripted.requests_asked(), 2)


# ── T26: status ─────────────────────────────────────────────────────────────────


class StatusTests(CliCase):
    async def make_threads(self) -> dict[str, str]:
        made = {}
        for name, server, lines in (("paused", FakeServer(), []),
                                    ("filed", FakeServer(), ["ok"]),
                                    ("rejected", FakeServer(), ["no", "budget"]),
                                    ("uncertain", FakeServer(write_mode="transport"), ["ok"])):
            _, out, _ = await run([], self.db, server=server, lines=lines)
            (made[name],) = threads_in(out)
        return made

    async def status(self, *argv):
        for p in no_runtime_resources():
            p.start()
        try:
            return await run(["--status", *argv], self.db, real_runtime=True)
        finally:
            mock.patch.stopall()

    async def test_t26_no_checkpoint_file(self):
        code, out, _ = await self.status()
        self.assertEqual(code, 0, out)
        self.assertIn("No saved threads.", out)
        self.assertFalse(self.db.exists())

    async def test_t26_every_state_and_its_next_action(self):
        t = await self.make_threads()
        code, out, _ = await self.status()
        self.assertEqual(code, 0, out)
        lines = {name: [line for line in out.splitlines() if tid in line] for name, tid in t.items()}
        self.assertIn(f"python supervisor.py --resume {t['paused']}", out)
        self.assertIn("awaiting approval", " ".join(lines["paused"]).lower())
        self.assertIn("filed", " ".join(lines["filed"]).lower())
        self.assertIn("rejected", " ".join(lines["rejected"]).lower())
        self.assertIn("uncertain", " ".join(lines["uncertain"]).lower())
        for name in ("filed", "rejected", "uncertain"):
            self.assertNotIn(f"--resume {t[name]}", out, name)

    async def test_t26_one_thread(self):
        t = await self.make_threads()
        code, out, _ = await self.status(t["paused"])
        self.assertEqual(code, 0, out)
        self.assertIn(f"python supervisor.py --resume {t['paused']}", out)
        self.assertIn("employee_id: E010", out)

        code, out, _ = await self.status(t["uncertain"])
        self.assertEqual(code, 0, out)
        self.assertIn("verify", out.lower())
        self.assertNotIn("--resume", out)

        code, out, _ = await self.status(t["filed"])
        self.assertIn("filed", out.lower())
        self.assertIn("AR101", out)
        self.assertNotIn("--resume", out)

        code, out, _ = await self.status("case-00000000000000-dead")
        self.assertEqual(code, 2, out)

    async def test_t26_resume_of_unknown_thread_creates_nothing(self):
        for p in no_runtime_resources():
            p.start()
        try:
            code, out, _ = await run(["--resume", "case-00000000000000-dead"], self.db, real_runtime=True)
        finally:
            mock.patch.stopall()
        self.assertEqual(code, 2, out)
        self.assertFalse(self.db.exists())


# ── T27: Ctrl-C and fresh processes (real asyncio.run) ─────────────────────────


def child(body: str, db: Path) -> list[str]:
    script = textwrap.dedent(f"""
        import asyncio, signal, sys, time
        sys.path[:0] = [{str(TESTS)!r}, {str(ROOT)!r}]
        import fakes, supervisor
        DB = {str(db)!r}
    """) + textwrap.dedent(body)
    return [sys.executable, "-c", script]


def run_child(body: str, db: Path, timeout: float = 15):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    done = subprocess.run(child(body, db), capture_output=True, text=True, encoding="utf-8",
                          timeout=timeout, env=env)
    return done, time.perf_counter() - started


class CtrlCAndRestartTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = Path(self._tmp.name) / "checkpoints.sqlite"

    def tearDown(self):
        self._tmp.cleanup()

    def test_t27_ctrl_c_at_the_approval_prompt_pauses(self):
        done, seconds = run_child("""
            def ask(payload):
                signal.raise_signal(signal.SIGINT)
                time.sleep(30)
            sys.exit(asyncio.run(supervisor.cli([], tools=fakes.FakeServer().tools(),
                                                model=fakes.make_model(), db_path=DB, ask=ask)))
        """, self.db)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Paused", done.stdout)
        self.assertLess(seconds, 10)

    def test_t27_ctrl_c_while_running_reports_and_exits_130(self):
        done, seconds = run_child("""
            async def stuck(messages, bound):
                signal.raise_signal(signal.SIGINT)
                await asyncio.sleep(30)
            sys.exit(asyncio.run(supervisor.cli([], tools=fakes.FakeServer().tools(),
                                                model=fakes.make_model(act=stuck), db_path=DB)))
        """, self.db)
        self.assertEqual(done.returncode, 130, done.stdout + done.stderr)
        self.assertIn("Interrupted.", done.stdout)
        self.assertIn("nothing was filed", done.stdout.lower())
        self.assertNotIn("Traceback", done.stderr)
        self.assertLess(seconds, 10)

    def test_t27_kill_at_the_prompt_then_resume_in_a_fresh_process(self):
        first = subprocess.Popen(child("""
            def ask(payload):
                print("WAITING-FOR-APPROVAL", flush=True)
                time.sleep(60)
            asyncio.run(supervisor.cli([], tools=fakes.FakeServer().tools(),
                                       model=fakes.make_model(), db_path=DB, ask=ask))
        """, self.db), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        seen = ""
        deadline = time.time() + 20
        while "WAITING-FOR-APPROVAL" not in seen and time.time() < deadline:
            line = first.stdout.readline()
            if not line:
                break
            seen += line
        first.kill()
        first.wait(10)
        first.stdout.close()
        self.assertIn("WAITING-FOR-APPROVAL", seen)
        (thread,) = threads_in(seen)

        done, _ = run_child(f"""
            server = fakes.FakeServer()
            code = asyncio.run(supervisor.cli(["--resume", {thread!r}], tools=server.tools(),
                                              model=fakes.make_model(), db_path=DB,
                                              ask=lambda payload: {{"approved": True}}))
            print("WRITES", len(server.writes), "ARGS", server.writes[0] if server.writes else None)
            sys.exit(code)
        """, self.db)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("WRITES 1", done.stdout)
        self.assertIn("'employee_id': 'E010'", done.stdout)


if __name__ == "__main__":
    unittest.main()
