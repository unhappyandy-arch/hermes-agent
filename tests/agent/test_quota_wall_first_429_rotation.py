"""方舟 429 额度墙「首败即轮换」（P2）回归测试。

缺陷：多 key 凭据池里当前 key 已撞额度墙（周额度用尽），Hermes 仍用同一把 key
重试一次（约 2–3 秒），二败才 bench 轮换 —— 对多 key 池没有信息增益
（2026-10-29 23:35 实测：attempt 1/3 → Retrying API call in 2.2s → 二败 → rotated）。

修复：仅「额度墙」+ 池 ``entries()>1`` + ``has_available()`` 时首败即 ``rotate_and_swap``；
突发限流（burst / throttle）、单 key 池、无可用条目保持原有「先重试」语义。
"""
import logging

import pytest

import run_agent  # 预热：_ra() 懒加载 run_agent 会在首次日志调用时探测 .git 标记，
# 放在收集阶段导入，避免 home_io_guard 在首个用例里把该探测判成「写真实 hermes home」。

from agent.agent_runtime_helpers import (
    _is_quota_wall_429,
    _recover_rate_limit,
    recover_with_credential_pool,
)
from agent.credential_pool import (
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
)
from agent.error_classifier import FailoverReason


# 方舟真实 429 文案（2026-10-29 实测抓取，原样保留中英文与双后缀时区）
ARK_WEEKLY = (
    "You have exceeded the weekly usage quota. It will reset at 2026-11-02 00:00:00 "
    "+0800 CST. We recommend upgrading your plan for more quota, or waiting for the reset. "
    "Request id: 0217932880991154e0d847e114bd906533eadfa024bfe6d63d98d"
)
# 生产 error_context 形状：extract_api_error_context 取 body.error.code 作 reason
ARK_CONTEXT = {"reason": "AccountQuotaExceeded", "message": ARK_WEEKLY}


class _Entry:
    def __init__(self, eid, key, status=None, label="g6"):
        self.id, self.runtime_api_key, self.last_status, self.label = eid, key, status, label


class _FakePool:
    """最小 fake pool：模拟首刀刚撞墙、当前条目尚未 bench 的那一刻。"""

    provider = "custom:custom"

    def __init__(self, entries=2, available=True):
        self._entries = [
            _Entry("e1", "k-g6"), _Entry("e2", "k-g2", label="g2"),
        ][:entries]
        self._available = available

    def entries(self):
        return self._entries

    def current(self):
        return self._entries[0]

    def has_available(self, **_kw):
        return self._available


def _call(pool, *, has_retried_429, error_context):
    """驱动 ``_recover_rate_limit``，记录 rotate 调用。返回 (recovered, retried, calls)。"""
    calls = []

    def fake_rotate(status, label):
        calls.append((status, label))
        return True

    recovered, retried = _recover_rate_limit(
        pool, has_retried_429=has_retried_429, error_context=error_context,
        api_key_hint="k-g6", credential_id="e1", rotate_and_swap=fake_rotate,
    )
    return recovered, retried, calls


# ── 缺陷 B 反转：额度墙 + 多 key 池 → 首败即轮换 ─────────────────────────────

def test_ark_weekly_quota_first_429_rotates_immediately():
    recovered, retried, calls = _call(
        _FakePool(), has_retried_429=False, error_context=ARK_CONTEXT,
    )
    assert recovered is True and retried is False
    assert calls == [(429, "rate limit, quota wall pre-rotate")]


def test_quota_wall_with_generic_reason_label_also_rotates():
    """分类器把 429 归成 rate_limit 时，仅凭文案也应触发首败即轮换。"""
    recovered, retried, calls = _call(
        _FakePool(), has_retried_429=False,
        error_context={"reason": "rate_limit", "message": ARK_WEEKLY},
    )
    assert recovered is True and retried is False and calls


def test_usage_limit_reached_still_rotates_on_first_failure():
    """既有语义保留：文案命中 usage-limit 词表时首败也轮换。"""
    recovered, retried, calls = _call(
        _FakePool(), has_retried_429=False,
        error_context={"reason": "usage_limit_reached", "message": "usage limit reached"},
    )
    assert recovered is True and retried is False and calls


def test_second_429_still_rotates():
    recovered, retried, calls = _call(
        _FakePool(), has_retried_429=True, error_context=ARK_CONTEXT,
    )
    assert recovered is True and retried is False
    assert calls == [(429, "rate limit")]


# ── 回归：突发限流 / 单 key 池 / 无可用条目 / 普通 429 保持「先重试」──────────

@pytest.mark.parametrize("burst_context", [
    {"reason": "RequestBurstTooFast", "message": "rate increased too quickly"},
    {"reason": "rate_limit", "message": "Too many requests, please retry after 2s"},
    {"reason": "rate_limit", "message": "Throttling: requests per minute exceeded"},
    {"reason": "rate_limit", "message": "quota-related burst spike"},  # 突发词表压过额度词表
])
def test_burst_throttle_keeps_retry_first_semantics(burst_context):
    recovered, retried, calls = _call(
        _FakePool(), has_retried_429=False, error_context=burst_context,
    )
    assert recovered is False and retried is True and not calls


def test_single_entry_pool_keeps_retry_first():
    """单 key 池：没有可换的条目，重试同一把 key 是唯一路径。"""
    recovered, retried, calls = _call(
        _FakePool(entries=1), has_retried_429=False, error_context=ARK_CONTEXT,
    )
    assert recovered is False and retried is True and not calls


def test_pool_without_available_entries_keeps_retry_first():
    """池里没有可用条目：轮换只会拿到 None，保持先重试、再走 fallback。"""
    recovered, retried, calls = _call(
        _FakePool(available=False), has_retried_429=False, error_context=ARK_CONTEXT,
    )
    assert recovered is False and retried is True and not calls


def test_generic_429_keeps_retry_first():
    recovered, retried, calls = _call(
        _FakePool(), has_retried_429=False, error_context={"message": "Error"},
    )
    assert recovered is False and retried is True and not calls


def test_pool_probe_failure_fails_closed_to_retry_first():
    class _NoProbePool(_FakePool):
        def has_available(self, **_kw):
            raise RuntimeError("availability probe failed")

    recovered, retried, calls = _call(
        _NoProbePool(), has_retried_429=False, error_context=ARK_CONTEXT,
    )
    assert recovered is False and retried is True and not calls


def test_rotation_yielding_no_entry_falls_back_to_retry():
    pool = _FakePool()

    def fake_rotate(status, label):
        return False

    recovered, retried = _recover_rate_limit(
        pool, has_retried_429=False, error_context=ARK_CONTEXT,
        api_key_hint="k-g6", credential_id="e1", rotate_and_swap=fake_rotate,
    )
    assert recovered is False and retried is True


# ── 判据单元 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("context,expected", [
    (ARK_CONTEXT, True),
    ({"reason": "rate_limit", "message": ARK_WEEKLY}, True),
    ({"message": "You have exceeded your current quota"}, True),
    ({"message": "额度已用尽，请稍后重试"}, True),
    ({"message": "Error"}, False),
    ({"reason": "RequestBurstTooFast", "message": "rate increased too quickly"}, False),
    (None, False),
    ({}, False),
])
def test_is_quota_wall_429_predicate(context, expected):
    assert _is_quota_wall_429(context) is expected


# ── 集成：真实 CredentialPool + recover_with_credential_pool 首败即换键 ─────

class _LiveAgent:
    """长期会话替身：真实池 + 真实恢复入口，不起 HTTP 客户端。"""

    provider = "deepseek"
    model = "deepseek-flash"
    base_url = "https://api.deepseek.com"

    def __init__(self, pool):
        self._credential_pool = pool
        first = pool.select()
        self._credential_pool_entry_id = first.id
        self.api_key = first.runtime_api_key

    def _swap_credential(self, entry):
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False


def _entry(entry_id, label, *, priority, token):
    return PooledCredential.from_dict("deepseek", {
        "id": entry_id, "label": label, "auth_type": "api_key", "priority": priority,
        "access_token": token, "source": "manual",
    })


def test_real_pool_rotates_on_first_quota_wall(caplog):
    pool = CredentialPool(provider="deepseek", entries=[
        _entry("dead0000", "g6-wall", priority=0, token="dead-key"),
        _entry("live0000", "g2-healthy", priority=1, token="live-key"),
    ])
    agent = _LiveAgent(pool)
    assert agent.api_key == "dead-key"

    with caplog.at_level(logging.INFO):
        recovered, retried = recover_with_credential_pool(
            agent, status_code=429, has_retried_429=False,
            classified_reason=FailoverReason.rate_limit, error_context=ARK_CONTEXT,
        )

    assert recovered is True and retried is False
    assert agent.api_key == "live-key"
    assert next(e for e in pool.entries() if e.id == "dead0000").last_status == STATUS_EXHAUSTED
    messages = [r.getMessage() for r in caplog.records]
    assert any("marking" in m and "exhausted" in m for m in messages), messages
    assert any("rotated to pool entry live0000" in m for m in messages), messages
    assert not any("Retrying API call in" in m for m in messages), messages


def test_real_pool_single_entry_keeps_retry_first(caplog):
    """回归：单 key 池行为不变——首败只置重试标志，不 bench、不轮换。"""
    pool = CredentialPool(provider="deepseek", entries=[
        _entry("only0000", "g6-wall", priority=0, token="dead-key"),
    ])
    agent = _LiveAgent(pool)

    with caplog.at_level(logging.INFO):
        recovered, retried = recover_with_credential_pool(
            agent, status_code=429, has_retried_429=False,
            classified_reason=FailoverReason.rate_limit, error_context=ARK_CONTEXT,
        )

    assert recovered is False and retried is True
    assert agent.api_key == "dead-key"
    assert next(e for e in pool.entries() if e.id == "only0000").last_status != STATUS_EXHAUSTED
    assert not any("marking" in r.getMessage() and "exhausted" in r.getMessage() for r in caplog.records)


# ── 端到端：真实 HTTP 调用（loopback 回放方舟 429 文案）─────────────────────

def test_real_http_call_rotates_on_first_ark_quota_wall(caplog):
    """真实 AIAgent + 真实 openai 客户端打 loopback：死 key 首次 429 即换键。

    回放的是方舟真实 body（AccountQuotaExceeded + 周墙文案），只把端点换成本机回环，
    不动生产 auth.json 池行、不烧真实额度。断言 agent.log 语义：首败即
    ``marking … exhausted`` / ``rotated to …``，且全程无 ``Retrying API call in``。
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen_auth: list[str] = []
    seen_paths: list[str] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):  # 静默：只在断言里看 hermes 自己的日志
            pass

        def _write_json(self, status: int, payload: dict):
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0)) or b"{}"))
            auth = self.headers.get("authorization", "")
            seen_auth.append(auth)
            seen_paths.append(self.path)
            if not self.path.endswith("/chat/completions"):
                # 探活类接口（/api/show 等）不属于 429 路径，回 404 让客户端自行降级
                self._write_json(404, {"error": {"message": "not found"}})
                return
            if auth.endswith("dead-key"):
                # 方舟真实 429 body（2026-10-29 抓取；无 retry-after、无 x-ratelimit-* 头）
                self._write_json(429, {"error": {
                    "code": "AccountQuotaExceeded", "type": "TooManyRequests", "message": ARK_WEEKLY,
                }})
                return
            chunk_base = {"id": "chatcmpl-x", "object": "chat.completion.chunk",
                          "created": 1, "model": "ark-code-latest"}
            if body.get("stream"):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                delta = {**chunk_base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "pong"}}]}
                stop = {**chunk_base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(delta)}\n\n".encode())
                self.wfile.write(f"data: {json.dumps(stop)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            self._write_json(200, {
                "id": "chatcmpl-x", "object": "chat.completion", "created": 1,
                "model": "ark-code-latest",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"},
                             "finish_reason": "stop"}],
            })

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        pool = CredentialPool(provider="deepseek", entries=[
            _entry("dead0000", "g6-wall", priority=0, token="dead-key"),
            _entry("live0000", "g2-healthy", priority=1, token="live-key"),
        ])
        with caplog.at_level(logging.INFO):
            agent = run_agent.AIAgent(
                api_key="dead-key", base_url=base_url, model="ark-code-latest", provider="deepseek",
                quiet_mode=True, skip_context_files=True, skip_memory=True,
                enabled_toolsets=[], max_iterations=1, credential_pool=pool,
            )
            agent._credential_pool_entry_id = "dead0000"
            result = agent.run_conversation("ping", conversation_history=[], task_id="t")
    finally:
        server.shutdown()
        server.server_close()

    messages = [r.getMessage() for r in caplog.records]
    chat_keys = [auth for auth, path in zip(seen_auth, seen_paths) if path.endswith("/chat/completions")]
    # 首败没有同键重试：chat 请求序列必须是「死 key 一次 → 活 key」，不多打一次墙
    assert chat_keys == ["Bearer dead-key", "Bearer live-key"], (chat_keys, seen_paths)
    assert not result["failed"], result
    assert agent.api_key == "live-key"
    assert any("marking" in m and "exhausted" in m for m in messages), messages
    assert any("rotated to pool entry live0000" in m for m in messages), messages
    assert not any("Retrying API call in" in m for m in messages), messages

