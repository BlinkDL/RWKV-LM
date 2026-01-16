# RWKV-8 "Heron" with ROSA (Rapid Online Suffix Automaton)

p.s. Can use float V in ROSA-QKV (still highly efficient). will try this soon.

### Community ROSA Projects

https://github.com/wjie98/rosa_soft (training ROSA)

https://github.com/zyaaa-ux/ROSA-Tuning (training ROSA)

https://github.com/bcml-ai/rosa-plus

https://github.com/x-0D/RASP

### Introducing ROSA 

<img src="../RWKV-8-ROSA.png">

251105_reverse_run.py - RWKV7+ROSA with 40K params (L2-D32) reversing 1-60 digits input with 99.8% digit accuracy:

<img width="1836" height="411" alt="image" src="https://github.com/user-attachments/assets/2af98f3e-721d-484f-8db3-ecd4ad777872" />

251024_rosaQKV_run.py for arithmetic demo (1M params can solve 40 digits plus/minus with 99% digit accuracy, without CoT):

<img width="1563" height="1008" alt="image" src="https://github.com/user-attachments/assets/12134e7c-85f3-4788-9664-8070152e7e72" />

251016_rosa_1bit_run.py for multi-layer ROSA demo:

<img width="1198" height="1198" alt="image" src="https://github.com/user-attachments/assets/ea2121b6-b571-4a95-9d5b-91e84c5d5e4a" />

251014_rosa_onlyemb_train.py will reach loss ~0.65

251014_rosa_1bit_train.py will reach loss ~0.4

251018_rosa_4bit_train.py will reach loss ~0.25

<img src="251014_rosa_1bit.png">
