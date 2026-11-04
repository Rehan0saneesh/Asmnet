"""
Micro Neural Network (pure Python) accelerated by Asmnet assembly math.

Uses ctypes to call the NASM x86-64 math library (System V AMD64 ABI) for the
heavy linear algebra: matmul, matadd, transpose. Activations, softmax, layer
norm, and the optimizer use Python's built-in math module (the asm library
provides integer sqrt and SSE add/sub/mul/div/neg/abs, but no exp/log/float-sqrt).

Same capabilities as the original python-testing/micro.py:
- character-level tokenizer
- token + positional embeddings
- layer normalization
- causal multi-head attention (single head)
- SwiGLU feed-forward block
- residual connections
- cross-entropy loss
- backpropagation
- Adam optimizer
- text generation from trained weights
"""

import ctypes
import math
import os
import random
import subprocess

# ---------------------------------------------------------------------------
# Load / build the assembly shared library
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD_DIR = os.path.join(ROOT, "build")
LIB_PATH = os.path.join(BUILD_DIR, "libasmnet.so")


def ensure_lib():
    if not os.path.exists(LIB_PATH):
        os.makedirs(BUILD_DIR, exist_ok=True)
        math_dir = os.path.join(ROOT, "math")
        asm_files = [
            os.path.join(math_dir, f)
            for f in os.listdir(math_dir)
            if f.endswith(".asm")
        ]
        objs = []
        for src in asm_files:
            name = os.path.splitext(os.path.basename(src))[0] + ".o"
            out = os.path.join(BUILD_DIR, name)
            subprocess.run(["nasm", "-f", "elf64", "-o", out, src], check=True)
            objs.append(out)
        subprocess.run(["gcc", "-shared", "-no-pie", "-o", LIB_PATH] + objs, check=True)
    return ctypes.CDLL(LIB_PATH)


lib = ensure_lib()

# Signatures (System V AMD64 ABI)
lib.m_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
lib.m_matmul.restype = None

lib.m_matadd.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
lib.m_matadd.restype = None

lib.m_matsub.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
lib.m_matsub.restype = None

lib.m_transpose.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
lib.m_transpose.restype = None

lib.m_fadd.argtypes = [ctypes.c_double, ctypes.c_double]
lib.m_fadd.restype = ctypes.c_double
lib.m_fsub.argtypes = [ctypes.c_double, ctypes.c_double]
lib.m_fsub.restype = ctypes.c_double
lib.m_fmul.argtypes = [ctypes.c_double, ctypes.c_double]
lib.m_fmul.restype = ctypes.c_double
lib.m_fdiv.argtypes = [ctypes.c_double, ctypes.c_double]
lib.m_fdiv.restype = ctypes.c_double
lib.m_fneg.argtypes = [ctypes.c_double]
lib.m_fneg.restype = ctypes.c_double
lib.m_fabs.argtypes = [ctypes.c_double]
lib.m_fabs.restype = ctypes.c_double


# ---------------------------------------------------------------------------
# Linear algebra helpers backed by assembly where available
# ---------------------------------------------------------------------------

def to_carray(matrix):
    flat = [float(v) for row in matrix for v in row]
    return (ctypes.c_double * len(flat))(*flat)


def to_list(arr, rows, cols):
    return [[arr[r * cols + c] for c in range(cols)] for r in range(rows)]


def matmul(A, B):
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    C = [[0.0] * cb for _ in range(ra)]
    cA = to_carray(A)
    cB = to_carray(B)
    cC = (ctypes.c_double * (ra * cb))(*([0.0] * (ra * cb)))
    lib.m_matmul(cA, ra, ca, cB, rb, cb, cC)
    return to_list(cC, ra, cb)


def matadd(A, B):
    r, c = len(A), len(A[0])
    cA = to_carray(A)
    cB = to_carray(B)
    cC = (ctypes.c_double * (r * c))(*([0.0] * (r * c)))
    lib.m_matadd(cA, r, c, cB, cC)
    return to_list(cC, r, c)


def matsub(A, B):
    r, c = len(A), len(A[0])
    cA = to_carray(A)
    cB = to_carray(B)
    cC = (ctypes.c_double * (r * c))(*([0.0] * (r * c)))
    lib.m_matsub(cA, r, c, cB, cC)
    return to_list(cC, r, c)


def transpose(A):
    r, c = len(A), len(A[0])
    cA = to_carray(A)
    cC = (ctypes.c_double * (c * r))(*([0.0] * (c * r)))
    lib.m_transpose(cA, r, c, cC)
    return to_list(cC, c, r)


def fadd(a, b):
    return lib.m_fadd(a, b)


def fsub(a, b):
    return lib.m_fsub(a, b)


def fmul(a, b):
    return lib.m_fmul(a, b)


def fdiv(a, b):
    return lib.m_fdiv(a, b)


def fneg(a):
    return lib.m_fneg(a)


def fabs(a):
    return lib.m_fabs(a)


# ---------------------------------------------------------------------------
# Neural network (same capabilities as python-testing/micro.py)
# ---------------------------------------------------------------------------

def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def softmax_row(row):
    m = max(row)
    exp_vals = [math.exp(v - m) for v in row]
    s = sum(exp_vals)
    return [e / s for e in exp_vals]


def silu(x):
    return x / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


def d_silu(x):
    s = 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))
    return s + x * s * (1.0 - s)


def layer_norm_forward(X, gamma, beta, eps=1e-5):
    N, d = len(X), len(X[0])
    X_norm = zeros(N, d)
    cache = []
    for i in range(N):
        mean = sum(X[i]) / d
        var = sum((X[i][j] - mean) ** 2 for j in range(d)) / d
        std = math.sqrt(var + eps)
        cache.append((X[i], mean, std))
        for j in range(d):
            X_norm[i][j] = gamma[j] * ((X[i][j] - mean) / std) + beta[j]
    return X_norm, cache


def layer_norm_backward(d_out, gamma, cache):
    N, d = len(d_out), len(d_out[0])
    dX = zeros(N, d)
    dgamma = [0.0] * d
    dbeta = [0.0] * d
    for i in range(N):
        X_row, mean, std = cache[i]
        t1, t2 = 0.0, 0.0
        for j in range(d):
            x_hat = (X_row[j] - mean) / std
            dgamma[j] += d_out[i][j] * x_hat
            dbeta[j] += d_out[i][j]
            t1 += d_out[i][j]
            t2 += d_out[i][j] * x_hat
        for j in range(d):
            x_hat = (X_row[j] - mean) / std
            dX[i][j] = (gamma[j] / std) * (d_out[i][j] - t1 / d - x_hat * (t2 / d))
    return dX, dgamma, dbeta


def attention_forward(X_norm, W_Q, W_K, W_V, W_O, d_k):
    N = len(X_norm)
    Q = matmul(X_norm, W_Q)
    K = matmul(X_norm, W_K)
    V = matmul(X_norm, W_V)
    scale = math.sqrt(d_k)
    S = matmul(Q, transpose(K))
    for i in range(N):
        for j in range(N):
            S[i][j] /= scale
            if j > i:
                S[i][j] = -1e9
    A = [softmax_row(S[i]) for i in range(N)]
    out = matmul(matmul(A, V), W_O)
    return out, (X_norm, Q, K, V, A, d_k)


def attention_backward(d_out, W_Q, W_K, W_V, W_O, cache):
    X_norm, Q, K, V, A, d_k = cache
    N = len(X_norm)
    context = matmul(A, V)
    dW_O = matmul(transpose(context), d_out)
    d_context = matmul(d_out, transpose(W_O))
    dV = matmul(transpose(A), d_context)
    dA = matmul(d_context, transpose(V))

    dS = zeros(N, N)
    scale = math.sqrt(d_k)
    for i in range(N):
        sum_dot = sum(dA[i][k] * A[i][k] for k in range(N))
        for j in range(i + 1):
            dS[i][j] = (A[i][j] * (dA[i][j] - sum_dot)) / scale

    dQ = matmul(dS, K)
    dK = matmul(transpose(dS), Q)
    dW_Q = matmul(transpose(X_norm), dQ)
    dW_K = matmul(transpose(X_norm), dK)
    dW_V = matmul(transpose(X_norm), dV)
    dX_norm = matadd(matadd(matmul(dQ, transpose(W_Q)), matmul(dK, transpose(W_K))), matmul(dV, transpose(W_V)))
    return dX_norm, {"W_Q": dW_Q, "W_K": dW_K, "W_V": dW_V, "W_O": dW_O}


def swiglu_forward(X_norm, W_gate, W_up, W_down):
    N, d_ff = len(X_norm), len(W_gate[0])
    Z_gate = matmul(X_norm, W_gate)
    H_up = matmul(X_norm, W_up)
    H_swiglu = zeros(N, d_ff)
    for i in range(N):
        for j in range(d_ff):
            H_swiglu[i][j] = silu(Z_gate[i][j]) * H_up[i][j]
    out = matmul(H_swiglu, W_down)
    return out, (X_norm, Z_gate, H_up, H_swiglu)


def swiglu_backward(d_out, W_gate, W_up, W_down, cache):
    X_norm, Z_gate, H_up, H_swiglu = cache
    N, d_ff = len(X_norm), len(Z_gate[0])
    dW_down = matmul(transpose(H_swiglu), d_out)
    dH_swiglu = matmul(d_out, transpose(W_down))
    dZ_gate = zeros(N, d_ff)
    dH_up = zeros(N, d_ff)
    for i in range(N):
        for j in range(d_ff):
            dH_up[i][j] = dH_swiglu[i][j] * silu(Z_gate[i][j])
            dZ_gate[i][j] = dH_swiglu[i][j] * H_up[i][j] * d_silu(Z_gate[i][j])
    dW_gate = matmul(transpose(X_norm), dZ_gate)
    dW_up = matmul(transpose(X_norm), dH_up)
    dX_norm = matadd(matmul(dZ_gate, transpose(W_gate)), matmul(dH_up, transpose(W_up)))
    return dX_norm, {"W_gate": dW_gate, "W_up": dW_up, "W_down": dW_down}


def forward_pass(token_ids, model):
    N, d = len(token_ids), len(model["W_E"][0])
    X = [[model["W_E"][token_ids[i]][j] + model["W_pos"][i][j] for j in range(d)] for i in range(N)]

    X_n1, c_ln1 = layer_norm_forward(X, model["ln1_g"], model["ln1_b"])
    attn_out, c_attn = attention_forward(X_n1, model["W_Q"], model["W_K"], model["W_V"], model["W_O"], model["d_k"])
    X_p = matadd(X, attn_out)

    X_n2, c_ln2 = layer_norm_forward(X_p, model["ln2_g"], model["ln2_b"])
    ffn_out, c_ffn = swiglu_forward(X_n2, model["W_gate"], model["W_up"], model["W_down"])
    X_b = matadd(X_p, ffn_out)

    X_f, c_lnf = layer_norm_forward(X_b, model["lnf_g"], model["lnf_b"])
    logits = matmul(X_f, model["W_U"])
    probs = [softmax_row(row) for row in logits]
    return logits, probs, (token_ids, c_ln1, c_attn, c_ln2, c_ffn, c_lnf, X_f)


def backward_pass(d_logits, model, cache):
    token_ids, c_ln1, c_attn, c_ln2, c_ffn, c_lnf, X_f = cache
    dW_U = matmul(transpose(X_f), d_logits)
    dX_f = matmul(d_logits, transpose(model["W_U"]))

    dX_b, dlnf_g, dlnf_b = layer_norm_backward(dX_f, model["lnf_g"], c_lnf)
    dX_n2, ffn_grads = swiglu_backward(dX_b, model["W_gate"], model["W_up"], model["W_down"], c_ffn)
    dX_p_ln2, dln2_g, dln2_b = layer_norm_backward(dX_n2, model["ln2_g"], c_ln2)
    dX_p = matadd(dX_b, dX_p_ln2)

    dX_n1, attn_grads = attention_backward(dX_p, model["W_Q"], model["W_K"], model["W_V"], model["W_O"], c_attn)
    dX_ln1, dln1_g, dln1_b = layer_norm_backward(dX_n1, model["ln1_g"], c_ln1)
    dX = matadd(dX_p, dX_ln1)

    N, d = len(token_ids), len(model["W_E"][0])
    dW_E = zeros(len(model["W_E"]), d)
    dW_pos = zeros(len(model["W_pos"]), d)
    for i in range(N):
        tid = token_ids[i]
        for j in range(d):
            dW_E[tid][j] += dX[i][j]
            dW_pos[i][j] += dX[i][j]

    return {
        "W_E": dW_E, "W_pos": dW_pos, "W_U": dW_U,
        "ln1_g": dln1_g, "ln1_b": dln1_b,
        "ln2_g": dln2_g, "ln2_b": dln2_b,
        "lnf_g": dlnf_g, "lnf_b": dlnf_b,
        **attn_grads, **ffn_grads
    }


def init_model(vocab_sz, max_len=16, d=8, d_k=8, d_ff=16):
    def rand_m(r, c):
        b = math.sqrt(2.0 / (r + c))
        return [[random.uniform(-b, b) for _ in range(c)] for _ in range(r)]
    return {
        "d_k": d_k,
        "W_E": rand_m(vocab_sz, d),
        "W_pos": rand_m(max_len, d),
        "ln1_g": [1.0] * d, "ln1_b": [0.0] * d,
        "W_Q": rand_m(d, d_k), "W_K": rand_m(d, d_k), "W_V": rand_m(d, d_k), "W_O": rand_m(d_k, d),
        "ln2_g": [1.0] * d, "ln2_b": [0.0] * d,
        "W_gate": rand_m(d, d_ff), "W_up": rand_m(d, d_ff), "W_down": rand_m(d_ff, d),
        "lnf_g": [1.0] * d, "lnf_b": [0.0] * d,
        "W_U": rand_m(d, vocab_sz)
    }


def train(model, data_ids, seq_len, epochs=80, lr=0.01):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = {k: zeros(len(model[k]), len(model[k][0])) if isinstance(model[k][0], list) else [0.0] * len(model[k]) for k in model if k != "d_k"}
    v = {k: zeros(len(model[k]), len(model[k][0])) if isinstance(model[k][0], list) else [0.0] * len(model[k]) for k in model if k != "d_k"}
    step = 0

    examples = [data_ids[i:i + seq_len + 1] for i in range(len(data_ids) - seq_len)]

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for ex in examples:
            inp, tgt = ex[:-1], ex[1:]
            N = len(inp)
            _, probs, cache = forward_pass(inp, model)

            vocab_sz = len(model["W_U"][0])
            d_logits = zeros(N, vocab_sz)
            loss = 0.0
            for t in range(N):
                target_id = tgt[t]
                p = max(probs[t][target_id], 1e-12)
                loss += -math.log(p)
                for k in range(vocab_sz):
                    d_logits[t][k] = (probs[t][k] - (1.0 if k == target_id else 0.0)) / N

            total_loss += loss / N
            grads = backward_pass(d_logits, model, cache)

            step += 1
            for k in grads:
                if isinstance(model[k][0], list):
                    for r in range(len(model[k])):
                        for c in range(len(model[k][0])):
                            g = grads[k][r][c]
                            m[k][r][c] = beta1 * m[k][r][c] + (1 - beta1) * g
                            v[k][r][c] = beta2 * v[k][r][c] + (1 - beta2) * (g ** 2)
                            m_hat = m[k][r][c] / (1 - beta1 ** step)
                            v_hat = v[k][r][c] / (1 - beta2 ** step)
                            model[k][r][c] -= lr * m_hat / (math.sqrt(v_hat) + eps)
                else:
                    for j in range(len(model[k])):
                        g = grads[k][j]
                        m[k][j] = beta1 * m[k][j] + (1 - beta1) * g
                        v[k][j] = beta2 * v[k][j] + (1 - beta2) * (g ** 2)
                        m_hat = m[k][j] / (1 - beta1 ** step)
                        v_hat = v[k][j] / (1 - beta2 ** step)
                        model[k][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Avg Loss: {total_loss / len(examples):.4f}")

    return examples


def generate(model, char_to_id, id_to_char, prompt_text, seq_len, max_new_tokens=10):
    input_ids = [char_to_id[c] for c in prompt_text if c in char_to_id]
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        ctx = generated[-seq_len:]
        _, probs, _ = forward_pass(ctx, model)
        last_p = probs[-1]
        next_id = max(range(len(last_p)), key=lambda k: last_p[k])
        generated.append(next_id)
    return "".join(id_to_char[idx] for idx in generated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    text_corpus = (
        "hello world\n"
        "low lower lowest\n"
        "high higher highest\n"
        "fast faster fastest\n"
        "slow slower slowest\n"
    )

    chars = sorted(list(set(text_corpus)))
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Dataset: {len(text_corpus)} characters, vocab ({vocab_size}): {repr(''.join(chars))}\n")

    seq_len = 8
    model = init_model(vocab_size, max_len=seq_len, d=12, d_k=12, d_ff=24)
    data_ids = [char_to_id[ch] for ch in text_corpus]

    train(model, data_ids, seq_len, epochs=80, lr=0.01)

    print("\n--- Model Inference ---")
    for test_prompt in ["low ", "fast", "high"]:
        print(f"Prompt: {repr(test_prompt)} -> Completion: {repr(generate(model, char_to_id, id_to_char, test_prompt, seq_len, 7))}")
