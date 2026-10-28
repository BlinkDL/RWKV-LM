#include "kernel_operator.h"
#include <cmath>

using namespace AscendC;

const int BUFFER_NUM = 2;

template <typename F>
class RWKV
{
public:
    __aicore__ inline RWKV() {}
    __aicore__ inline void init(const int B, const int T, const int C, const int H,
                                GM_ADDR state,
                                GM_ADDR r, GM_ADDR w, GM_ADDR k,
                                GM_ADDR v, GM_ADDR a, GM_ADDR b,
                                GM_ADDR y);
    __aicore__ inline void process();

private:
    __aicore__ inline void copyIn(size_t t);
    __aicore__ inline void compute(size_t t);
    __aicore__ inline void copyOut(size_t t);

private:
    // gm input
    GlobalTensor<F> _r, _w, _k, _v, _a, _b;
    // gm output
    GlobalTensor<F> _y;
    // gm state
    GlobalTensor<float> _state;

    // que input
    TQue<TPosition::VECIN, BUFFER_NUM> _que_r, _que_w, _que_k, _que_v, _que_a, _que_b;
    // que output
    TQue<TPosition::VECOUT, BUFFER_NUM> _que_y;

    // que state
    TBuf<TPosition::VECCALC> _buf_state;

    TPipe _pipe;
    int _B, _T, _C, _H, _N;
    int _e; // batch_idx
    int _h; // head_idx;
};

template <typename F>
__aicore__ inline void RWKV<F>::init(const int B, const int T, const int C, const int H,
                                     GM_ADDR state,
                                     GM_ADDR r, GM_ADDR w, GM_ADDR k,
                                     GM_ADDR v, GM_ADDR a, GM_ADDR b,
                                     GM_ADDR y)
{
    _B = B;
    _T = T;
    _C = C;
    _H = H;
    _N = _C / _H;

    _e = GetBlockIdx() / _H; // batch_idx
    _h = GetBlockIdx() % _H; // head_idx

    int rwkv_bias = _e * _T * _C + _h * _N;
    int state_bias = _e * _N * _N * _H + _h * _N * _N;

    // set gm
    _state.SetGlobalBuffer((__gm__ float *)state + state_bias);
    _r.SetGlobalBuffer((__gm__ F *)r + rwkv_bias);
    _w.SetGlobalBuffer((__gm__ F *)w + rwkv_bias);
    _k.SetGlobalBuffer((__gm__ F *)k + rwkv_bias);
    _v.SetGlobalBuffer((__gm__ F *)v + rwkv_bias);
    _a.SetGlobalBuffer((__gm__ F *)a + rwkv_bias);
    _b.SetGlobalBuffer((__gm__ F *)b + rwkv_bias);
    _y.SetGlobalBuffer((__gm__ F *)y + rwkv_bias);

    // init que
    _pipe.InitBuffer(_que_r, BUFFER_NUM, _N * sizeof(F));
    _pipe.InitBuffer(_que_w, BUFFER_NUM, _N * sizeof(F));
    _pipe.InitBuffer(_que_k, BUFFER_NUM, _N * sizeof(F));
    _pipe.InitBuffer(_que_v, BUFFER_NUM, _N * sizeof(F));
    _pipe.InitBuffer(_que_a, BUFFER_NUM, _N * sizeof(F));
    _pipe.InitBuffer(_que_b, BUFFER_NUM, _N * sizeof(F));
    _pipe.InitBuffer(_que_y, BUFFER_NUM, _N * sizeof(F));

    // init buf
    _pipe.InitBuffer(_buf_state, _N * _N * sizeof(float));
}

template <typename F>
__aicore__ inline void RWKV<F>::copyIn(size_t t)
{
    LocalTensor<F> r = _que_r.AllocTensor<F>();
    LocalTensor<F> w = _que_w.AllocTensor<F>();
    LocalTensor<F> k = _que_k.AllocTensor<F>();
    LocalTensor<F> v = _que_v.AllocTensor<F>();
    LocalTensor<F> a = _que_a.AllocTensor<F>();
    LocalTensor<F> b = _que_b.AllocTensor<F>();

    DataCopy(r, _r[t * _C], _N);
    DataCopy(w, _w[t * _C], _N);
    DataCopy(k, _k[t * _C], _N);
    DataCopy(v, _v[t * _C], _N);
    DataCopy(a, _a[t * _C], _N);
    DataCopy(b, _b[t * _C], _N);

    _que_r.EnQue(r);
    _que_w.EnQue(w);
    _que_k.EnQue(k);
    _que_v.EnQue(v);
    _que_a.EnQue(a);
    _que_b.EnQue(b);
}

template <typename F>
__aicore__ inline void RWKV<F>::compute(size_t t)
{
    // get input
    LocalTensor<F> r = _que_r.DeQue<F>();
    LocalTensor<F> w = _que_w.DeQue<F>();
    LocalTensor<F> k = _que_k.DeQue<F>();
    LocalTensor<F> v = _que_v.DeQue<F>();
    LocalTensor<F> a = _que_a.DeQue<F>();
    LocalTensor<F> b = _que_b.DeQue<F>();

    // get state
    LocalTensor<float> state = _buf_state.Get<float>();

    // get output
    LocalTensor<F> y = _que_y.AllocTensor<F>();

    // compute w
    Exp<F>(w, w, _N);
    Muls<F>(w, w, F(-1.0), _N);
    Exp<F>(w, w, _N);

    // compute
    for (int i = 0; i < _N; i++)
    {
        float sa = 0;
        // PipeBarrier<PIPE_V>();
#pragma unroll
        for (int j = 0; j < _N; j++)
        {
            sa += float(a(j)) * state(i * _N + j);
        }

        float vv = float(v(i));
        float yy = 0;
#pragma unroll
        for (int j = 0; j < _N; j++)
        {
            // __ubuf__ float *s = &state(i * _N + j);
            // *s = *s * float(w(j)) + float(k(j)) * vv + sa * float(b(j));
            // yy += *s * float(r(j));
            float s = state(i * _N + j);
            s = s * float(w(j)) + float(k(j)) * vv + sa * float(b(j));
            yy += s * float(r(j));
            state.SetValue(i * _N + j, s);
        }
        // PipeBarrier<PIPE_V>();
        y(i) = F(yy);
    }

    _que_r.FreeTensor(r);
    _que_w.FreeTensor(w);
    _que_k.FreeTensor(k);
    _que_v.FreeTensor(v);
    _que_a.FreeTensor(a);
    _que_b.FreeTensor(b);
    _que_y.EnQue(y);
}

template <typename F>
__aicore__ inline void RWKV<F>::copyOut(size_t t)
{
    LocalTensor<F> y = _que_y.DeQue<F>();
    DataCopy(_y[t * _C], y, _N);
    _que_y.FreeTensor(y);
}

template <typename F>
__aicore__ inline void RWKV<F>::process()
{
    LocalTensor<float> state_local = _buf_state.Get<float>();
    DataCopy(state_local, _state, _N * _N);
    for (int t = 0; t < _T; t++)
    {
        copyIn(t);
        compute(t);
        copyOut(t);
    }
    DataCopy(_state, state_local, _N * _N);
}

extern "C" __global__ __aicore__ void kernel_forward(int B, int T, int C, int H,
                                                     GM_ADDR state,
                                                     GM_ADDR r,
                                                     GM_ADDR w,
                                                     GM_ADDR k,
                                                     GM_ADDR v,
                                                     GM_ADDR a,
                                                     GM_ADDR b,
                                                     GM_ADDR y)
{
    RWKV<half> rwkv;
    rwkv.init(B, T, C, H, state, r, w, k, v, a, b, y);
    rwkv.process();
}

void ascend_forward(int B, int T, int C, int H, void *state, void *r, void *w, void *k, void *v, void *a, void *b, void *y, void* stream)
{
    assert(C % H == 0);
    assert(B == 1); // only for B=1
    kernel_forward<<<B * H, nullptr, stream>>>(B, T, C, H, state, r, w, k, v, a, b, y);
}
