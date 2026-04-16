# Improving RNNs (RWKV-8 and beyond)

Here I will show a framework to improve current RNNs.

## 1. Larger State

This includes larger heads size, larger inner model, hybrid models, ...

For RNNs, larger state => better performance, but bsz will be limited. And I belive massive parallel prefilling+decoding (large bsz, multi-agent) is the future.

From my view, there is a ladder of states: scalar state => vector state => matrix state (most current RNNs) => tensor state (a few papers tried this) => function state (attention is actually here, because it's kernel regression) => functional state => functor state => higher functor state => ...

Can certainly go beyond linear algebra: group, lie group, differential geometry, function space, category and higher categories, ... and only limited by efficient hardware implementation.

Indeed, new hardware (analog, quantum, ...) can change space and time complexity of some items in the ladder, and we are very far from endgame.

Example of a pratical step. Tensor states can be efficient, if only used in some heads, such as the slowest-decaying head. Use the better sum_{i,j} a[i] b[j] s[i,j,n] instead of the common idea sum_{i,j} a[i] a[j] s[i,j,n], and a 64x64x64 state can be a good starting point.

Note RWKV-4 has particular small states, and good for improvements.

## 2. Smaller State

This includes various tricks: sparse state, structured state, shared state, compressed state, low-rank state, quantized state, ... which can be found in various shrink-kv-cache papers too.

From my view, we can consider 6 dimensions: B (bsz), T (ctxlen), H (head), N (headsz), L (layer), Q (bits).

RNN statesz = f(B,H,N,L,Q). Transformer statesz = f(B,T,H,N,L,Q).

Can apply any trick to any dimension. Good for bingo.

Example:

H + sparse: use a router to select head.

N + sparse: use a router to select state inside a head. Larger state, similar I/O.

L + share: just like how a few papers proposed sharing kv cache between layers.

L + sparse: no need to go through all layers for all tokens.

T + compress: such as, compressing tokens into super-tokens, and can use raw bytes without tokenizer. Or, different ctxlen in different layers, such as T T/2 T T/2, T T/2 T/4 etc, and can restrict this to the hybrid attention part too.

Plenty of possiblities for each X + Y comination, and good for NAS.

## 3. Mixed State

Mixing state between heads. Mixing state between layers. These are expensive (when doing bwd). Can do them periodically, or when neccesary. Can do them at readout (cheaper).

Mixing state of the last layer of token n, with the state of the first layer of token n+1. A depth-L model becomes a depth-2L model after a step of this, and still efficiently trainable.

## 4. Fancy State Evolution

Example: Let A = evolution matrix. Try exp(sA)-1, 1/(1-sA), etc. with trainable dynamic s.

Example: DeltaProduct, fancy inner optimizers, fancy inner models.

These are all beneficial, and the question is {depth-L1 model with fancy state evolution} vs {depth-L2 model with simple state evolution} where L2 > L1 and speed-matched.

### Conclusion: we have room for 100 architecture papers here.

There are a number of more advanced methods beyond these, which I am exploring for RWKV-8.
