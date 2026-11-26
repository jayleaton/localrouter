// Ours: a host test of the HIP WMMA GEMM's element map (kernels/hip/common/wmma_map.h), no GPU and no ROCm: `zig build
// test` compiles it with Zig's clang. It runs wmma_gemm.h's loop structure on the CPU (blocks, waves, lanes, the two LDS
// stages, the zero-filled copies, the skipped accumulator blocks) through the same map functions the kernels use, with a
// stand-in for v_wmma (any fixed function of the accumulator and the 16 products), and checks for every tile shape the
// kernels instantiate and shapes at and around the tile edges:
//   - every element of C inside M x N is written exactly once, and nothing outside it;
//   - each element equals the reference: acc = 0, then the k16 chunks 0 .. 2 ceil(K / 32) - 1 of its row of A and row of
//     B in order, zeros past K (so the four f16 kernels and the bf16 one agree bit for bit);
//   - a row's bits do not depend on M: the first rows of an (M, N, K) product equal the product of those rows alone.
// What it cannot check is the hardware: the operand and result lane layouts (wmma_map.h's comment) and the instruction's
// own order are checked on gfx1151 by G0 (docs/dev/ROCM.md section 5).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../common/wmma_map.h"

using namespace lr_wmma;

namespace {

int failures = 0;

#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            if (failures < 20) {                 \
                std::fprintf(stderr, __VA_ARGS__); \
                std::fprintf(stderr, "\n");      \
            }                                    \
            ++failures;                          \
        }                                        \
    } while (0)

// The stand-in for one v_wmma k16 step of one element: a fixed order, so any reordering of chunks or operands shows.
float chunk(float acc, const float* a, const float* b) {
    float s = 0.0f;
    for (int k = 0; k < 16; ++k) s = std::fmaf(a[k], b[k], s);
    return acc + s * 1.0000001f;
}

struct Mat {
    int rows, cols;
    std::vector<float> v;
    float at(int r, int c) const { return v[(size_t)r * cols + c]; }
};

uint32_t rng = 12345;
float rnd() {  // values with 8 significant bits, as bf16 holds, spread over a few binades
    rng = rng * 1664525u + 1013904223u;
    const int m = (int)(rng >> 24) - 128;
    const int e = (int)((rng >> 8) & 7) - 4;
    return std::ldexp((float)m, e - 7);
}

Mat random(int rows, int cols) {
    Mat m{rows, cols, std::vector<float>((size_t)rows * cols)};
    for (float& x : m.v) x = rnd();
    return m;
}

float reference(const Mat& A, const Mat& B, int row, int col, int K) {
    float acc = 0.0f;
    for (int c = 0; c < 2 * k_tiles(K); ++c) {
        float a[16], b[16];
        for (int k = 0; k < 16; ++k) {
            const int kk = 16 * c + k;
            a[k] = kk < K ? A.at(row, kk) : 0.0f;
            b[k] = kk < K ? B.at(col, kk) : 0.0f;
        }
        acc = chunk(acc, a, b);
    }
    return acc;
}

// wmma_gemm.h's gemm<Cf> on the CPU. Returns C (M x N) and how often each element was written.
template <class Cf>
void emulate(const Mat& A, const Mat& B, int M, int N, int K, std::vector<float>& C, std::vector<int>& writes) {
    C.assign((size_t)M * N, NAN);
    writes.assign((size_t)M * N, 0);
    const int gm = (M + Cf::BM - 1) / Cf::BM, gn = (N + Cf::BN - 1) / Cf::BN;
    const int gx = Cf::M_FAST ? gm : gn, gy = Cf::M_FAST ? gn : gm;
    std::vector<float> sa(2 * Cf::BM * Cf::LD), sb(2 * Cf::BN * Cf::LD);
    for (int by = 0; by < gy; ++by)
        for (int bx = 0; bx < gx; ++bx) {
            const int m0 = tile_m0<Cf>(bx, by), n0 = tile_n0<Cf>(bx, by);
            // acc[wave][i][j][lane][v]
            const int waves = Cf::WM * Cf::WN;
            std::vector<float> acc((size_t)waves * Cf::MI * Cf::NJ * 32 * 8, 0.0f);
            auto A_ = [&](int w, int i, int j, int l, int v) -> float& { return acc[((((size_t)w * Cf::MI + i) * Cf::NJ + j) * 32 + l) * 8 + v]; };
            auto stage = [&](int s, int k0) {  // load_tile + store_tile of every thread
                for (int t = 0; t < Cf::THREADS; ++t) {
                    for (int i = 0; i < Cf::A_CH; ++i) {
                        const int e = t + i * Cf::THREADS, r = stage_row(e), c = stage_col(e), k = k0 + c;
                        for (int x = 0; x < CHUNK; ++x) sa[s * Cf::BM * Cf::LD + r * Cf::LD + c + x] = (k < K && m0 + r < M) ? A.at(m0 + r, k + x) : 0.0f;
                    }
                    for (int i = 0; i < Cf::B_CH; ++i) {
                        const int e = t + i * Cf::THREADS, r = stage_row(e), c = stage_col(e), k = k0 + c;
                        for (int x = 0; x < CHUNK; ++x) sb[s * Cf::BN * Cf::LD + r * Cf::LD + c + x] = (k < K && n0 + r < N) ? B.at(n0 + r, k + x) : 0.0f;
                    }
                }
            };
            std::fill(sa.begin(), sa.end(), NAN);  // a slot never written shows as NaN
            std::fill(sb.begin(), sb.end(), NAN);
            stage(0, 0);
            const int tiles = k_tiles(K);
            for (int t = 0; t < tiles; ++t) {
                const int s = t & 1;
                for (int w = 0; w < waves; ++w) {
                    const int wm = wave_m<Cf>(32 * w), wn = wave_n<Cf>(32 * w);
                    for (int kk = 0; kk < BK; kk += 16)
                        for (int i = 0; i < Cf::MI; ++i) {
                            if (!(m0 + wm * Cf::TM + i * 16 < M)) continue;
                            for (int j = 0; j < Cf::NJ; ++j) {
                                if (!(n0 + wn * Cf::TN + j * 16 < N)) continue;
                                // the instruction: lane l's A fragment is row l % 16 of the 16 x 16 A, its B fragment column l % 16
                                float a[16][16], b[16][16];
                                for (int l = 0; l < 16; ++l)
                                    for (int k = 0; k < 16; ++k) {
                                        a[l][k] = sa[s * Cf::BM * Cf::LD + frag_a_row<Cf>(wm, i, l) * Cf::LD + kk + k];
                                        b[l][k] = sb[s * Cf::BN * Cf::LD + frag_b_row<Cf>(wn, j, l) * Cf::LD + kk + k];
                                    }
                                for (int l = 0; l < 32; ++l)
                                    for (int v = 0; v < 8; ++v) {
                                        const int r = 2 * v + l / 16, c = l % 16;  // D's element in lane l, register v
                                        float& d = A_(w, i, j, l, v);
                                        d = chunk(d, a[r], b[c]);
                                    }
                            }
                        }
                }
                if (t + 1 < tiles) stage(s ^ 1, (t + 1) * BK);
            }
            for (int w = 0; w < waves; ++w) {
                const int wm = wave_m<Cf>(32 * w), wn = wave_n<Cf>(32 * w);
                for (int i = 0; i < Cf::MI; ++i)
                    for (int j = 0; j < Cf::NJ; ++j) {
                        if (!(m0 + wm * Cf::TM + i * 16 < M) || !(n0 + wn * Cf::TN + j * 16 < N)) continue;
                        for (int l = 0; l < 32; ++l) {
                            const int col = n0 + out_col<Cf>(wn, j, l);
                            for (int v = 0; v < 8; ++v) {
                                const int row = m0 + out_row<Cf>(wm, i, v, l);
                                if (row < M && col < N) {
                                    C[(size_t)row * N + col] = A_(w, i, j, l, v);
                                    ++writes[(size_t)row * N + col];
                                }
                            }
                        }
                    }
            }
        }
}

template <class Cf>
void check(const char* name, int M, int N, int K) {
    const Mat A = random(M, K), B = random(N, K);
    std::vector<float> C;
    std::vector<int> w;
    emulate<Cf>(A, B, M, N, K, C, w);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            const size_t e = (size_t)r * N + c;
            CHECK(w[e] == 1, "%s M %d N %d K %d: C[%d, %d] written %d times", name, M, N, K, r, c, w[e]);
            const float want = reference(A, B, r, c, K);
            CHECK(std::memcmp(&C[e], &want, 4) == 0, "%s M %d N %d K %d: C[%d, %d] = %a, reference %a", name, M, N, K, r, c, C[e], want);
        }
    // row independence: the first rows alone
    const int m2 = M > 1 ? M / 2 + 1 : 1;
    Mat A2{m2, K, std::vector<float>(A.v.begin(), A.v.begin() + (size_t)m2 * K)};
    std::vector<float> C2;
    emulate<Cf>(A2, B, m2, N, K, C2, w);
    CHECK(std::memcmp(C2.data(), C.data(), (size_t)m2 * N * 4) == 0, "%s M %d N %d K %d: rows depend on M", name, M, N, K);
}

// Every tile shape the kernels instantiate (gemm.hip, gemm_f16.hip, attention.hip's are not GEMM tiles).
using Bf16 = Cfg<128, 128, 2, 4, false>;
using F16Ref = Cfg<128, 128, 2, 4, false>;
using F16Wide = Cfg<256, 128, 4, 2, true>;
using F16Std = Cfg<128, 128, 2, 4, true>;
using F16Narrow = Cfg<64, 64, 2, 2, true>;

template <class Cf>
void shapes(const char* name) {
    const int s[][3] = {{1, 1, 8}, {16, 16, 16}, {17, 33, 40}, {64, 64, 32}, {65, 63, 72}, {128, 128, 64}, {129, 130, 24}, {257, 129, 104}, {300, 70, 8}};
    for (const auto& x : s) check<Cf>(name, x[0], x[1], x[2]);
}

}  // namespace

int main() {
    static_assert(F16Wide::SMEM == 61440 && F16Std::SMEM == 40960 && F16Narrow::SMEM == 20480 && Bf16::SMEM == 40960, "LDS a block");
    static_assert(Bf16::THREADS == 256 && F16Wide::THREADS == 256 && F16Narrow::THREADS == 128, "the CUDA kernels' block sizes");
    shapes<Bf16>("bf16");
    shapes<F16Ref>("f16_ref");
    shapes<F16Wide>("f16");
    shapes<F16Std>("f16_s");
    shapes<F16Narrow>("f16_n");
    if (failures) {
        std::fprintf(stderr, "wmma_map_test: %d failures\n", failures);
        return 1;
    }
    std::printf("wmma_map_test: ok\n");
    return 0;
}
