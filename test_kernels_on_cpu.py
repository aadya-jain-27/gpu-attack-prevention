"""Compiles the CUDA kernels as plain C++ (one loop iteration per GPU thread) and checks them
against the CPU pipeline on random fuzz traffic. Tests the kernel logic without needing a GPU.
Run: python3 tests/test_kernels_on_cpu.py"""
import ctypes, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
code = open(os.path.join(HERE, "..", "gpu_ids.py")).read()

# Load rules, automaton and CPU baseline from the main script without running the benchmarks
ns = {}
exec(code[:code.index("# %% [markdown]\n# ## 3.")].replace("import pycuda.autoinit", "raise ImportError('cpu test')"), ns)
exec(code[code.index("LOWER_RULES ="):code.index("# %% [markdown]\n# ## 7.")], ns)
KERNELS = code.split('KERNELS = r"""')[1].split('"""')[0]

SHIM = r"""
#include <stddef.h>
struct dim3_ { int x; };
static dim3_ blockIdx, blockDim, threadIdx;
#define __global__ static
#define __device__ static
#define __forceinline__ inline
#define __restrict__ __restrict
static int atomicAdd(int* a, int v) { int o = *a; *a += v; return o; }
"""
LAUNCH = r"""
#define LAUNCH(n) for (int t = 0; t < (n); t++) { blockDim.x = 256; blockIdx.x = t / 256; threadIdx.x = t % 256;
extern "C" {
void run_row(const unsigned char* p, const int* l, int ml, const int* d, const unsigned long long* o, unsigned long long* r, int n)
{ LAUNCH(n + 100) ac_match_rowmajor(p, l, ml, d, o, r, n); } }
void run_col(const unsigned char* p, const int* l, int ml, const int* d, const unsigned long long* o, unsigned long long* r, int n)
{ LAUNCH(n + 100) ac_match_colmajor(p, l, ml, d, o, r, n); } }
void run_count(const int* s, const int* g, int* pc, int* lc, int n) { LAUNCH(n) count_sources(s, g, pc, lc, n); } }
void run_policy(const unsigned long long* r, const int* s, const int* pc, const int* lc, const int* pb,
                int fl, int ll, int* b, unsigned char* v, int n)
{ LAUNCH(n) apply_policy(r, s, pc, lc, pb, fl, ll, b, v, n); } }
}
"""
tmp = tempfile.mkdtemp()
open(f"{tmp}/k.cpp", "w").write(SHIM + KERNELS + LAUNCH)
subprocess.run(["clang++", "-O2", "-Wall", "-shared", "-fPIC", "-o", f"{tmp}/k.so", f"{tmp}/k.cpp"], check=True)
lib = ctypes.CDLL(f"{tmp}/k.so")
P = lambda a: a.ctypes.data_as(ctypes.c_void_p)

RULES, DELTA, OUT_MASK, ML = ns["RULES"], ns["DELTA"], ns["OUT_MASK"], ns["MAX_LEN"]
rng = np.random.default_rng(7)

# Fuzz traffic: random bytes drawn from the characters used in the rules, with full rules and
# rule prefixes spliced in, in mixed case. This stresses overlapping and nested patterns.
alphabet = np.frombuffer(b"".join(p for p, _ in RULES), dtype=np.uint8)
n, n_src = 20000, 500
mat = rng.choice(alphabet, size=(n, ML)).astype(np.uint8)
for i in range(n):
    for _ in range(rng.integers(0, 4)):
        pat = RULES[rng.integers(len(RULES))][0]
        if rng.random() < 0.5:
            pat = pat[: rng.integers(1, len(pat) + 1)]
        if rng.random() < 0.5:
            pat = pat.upper()
        pos = rng.integers(0, ML - len(pat) + 1)
        mat[i, pos:pos + len(pat)] = np.frombuffer(pat, np.uint8)
lens = rng.integers(0, ML + 1, size=n).astype(np.int32)
src_ids = rng.integers(0, n_src, size=n).astype(np.int32)
login = (rng.random(n) < 0.1).astype(np.int32)
prev = (rng.random(n_src) < 0.05).astype(np.int32)
FL, LL = 45, 5

c_sig, c_ver, c_blk, c_pc, _ = ns["cpu_pipeline"](mat, lens, src_ids, login, n_src, FL, LL, prev)

delta = np.ascontiguousarray(DELTA.ravel())
for name, fn, data in [("row", lib.run_row, mat), ("col", lib.run_col, np.ascontiguousarray(mat.T))]:
    res = np.zeros(n, np.uint64)
    fn(P(data), P(lens), ML, P(delta), P(OUT_MASK), P(res), n)
    bad = np.flatnonzero(res != c_sig)
    print(f"{name}-major kernel: {n - len(bad)}/{n} packets match CPU, {int((res != 0).sum())} with hits")
    assert len(bad) == 0, bad[:10]

pc = np.zeros(n_src, np.int32); lc = np.zeros(n_src, np.int32)
lib.run_count(P(src_ids), P(login), P(pc), P(lc), n)
blk = np.zeros(n_src, np.int32); ver = np.zeros(n, np.uint8)
lib.run_policy(P(res), P(src_ids), P(pc), P(lc), P(prev), FL, LL, P(blk), P(ver), n)
assert np.array_equal(pc, c_pc) and np.array_equal(blk, c_blk) and np.array_equal(ver, c_ver)
print("count + policy kernels match CPU; verdicts:", np.bincount(ver, minlength=3).tolist(),
      "| blocked sources:", int(blk.sum()))
print("ALL OK")
