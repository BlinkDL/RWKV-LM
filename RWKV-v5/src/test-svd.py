# from .svd import recover_save_0_1B
import svd

svd.recover_save("/data/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/L12-D768-F16-x052att/rwkv-final",
                      "/tmp/myrecover", nlayers=12, nembd=768)