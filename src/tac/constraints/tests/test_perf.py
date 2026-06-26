import torch
from tac.core.state import State
from tac.constraints.length import Length, VectorizedLength
import time


def test_vectorized_length_parity():
    b = 1
    N = 100
    N_pairs = 500

    points = torch.randn(N, 3)
    state_1 = State(points, batch_size=b)
    state_2 = State(torch.clone(points), batch_size=b)
    pairs = torch.randint(0, N, (N_pairs, 2))

    vectorized_length = VectorizedLength(state_1, pairs[:, 0], pairs[:, 1], 1.0, 0.1)
    lengths = []
    for i in range(N_pairs):
        lengths.append(Length(state_2, pairs[i, 0], pairs[i, 1], 1.0, 0.1))

    s = time.perf_counter()
    for _ in range(10):
        lam_v = vectorized_length.apply(torch.zeros(b, N_pairs, 1), 0.1)
    e = time.perf_counter()
    v_perf = (e - s) / 10
    print(f"Time taken for vectorized length: {v_perf} seconds")

    s = time.perf_counter()
    for _ in range(10):
        for i in range(N_pairs):
            lengths[i].apply(torch.zeros(b, 1), 0.1)
    e = time.perf_counter()
    s_perf = (e - s) / 10
    print(f"Time taken for sequential lengths: {s_perf} seconds")

    print(f"Speedup: {s_perf / v_perf}")
    assert s_perf / v_perf > 100

