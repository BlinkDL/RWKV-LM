# 1. 编译算子
cd mkdir build
cd build
cmake ..
make 

# 2. 测试算子
python test_rwkv.py

# 3. 运行模型
cd ..
python rwkv_v7_demo_fast_npu.py