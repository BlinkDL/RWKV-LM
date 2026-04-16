I added 2T tokens to RWKV-6 "World-3" 7.6B and trained RWKV-7 "G0" 7.2B, likely the strongest pure RNN ever existed.

Download model: https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/rwkv7-g0-7.2b-20250722-ctx4096.pth

How to run: use https://github.com/josStorer/RWKV-Runner/releases for reference

This is a pretrained base model, no SFT, no RL, but with plenty of instruction/chat/reasoning data (all from HF) added to corpus.

I find RWKV-7 "G0" 7.2B can correct its errors in reasoning, and solve modified math problems. So **RNN + Pretrain + Scaling is all you need**.

Feel free to try temp=0 topp=0 penalty=0 0 (greedy decoding, usually ok, but sometimes stuck) or temp=0.3 topp=0.3 penalty=0 0 (probably better) etc.

Let's see how it works. First question:

<img width="965" height="1363" alt="image" src="https://github.com/user-attachments/assets/3bb43da3-f0db-46f8-8d3a-6703326f6642" />

---

Then I change the words. Now it's using another method:

<img width="648" height="1358" alt="image" src="https://github.com/user-attachments/assets/b83bb2eb-b651-4bee-9690-b766548c1ffc" />

---

A modified problem (99 => 99.1). It can correct its error.

<img width="964" height="1372" alt="image" src="https://github.com/user-attachments/assets/28dc2901-cddd-439a-a92c-49828f1325c8" />

---

Here I change "1^" to "i^" to increase difficulty.

<img width="642" height="1367" alt="image" src="https://github.com/user-attachments/assets/9814c3cf-55b1-484b-b670-8ac97a04ad7f" />

---

Here I change "4^x" to "8^x" to increase difficulty.

<img width="645" height="1367" alt="image" src="https://github.com/user-attachments/assets/98bb4c20-0d7d-41b1-ba1b-60223b52e5c2" />

---

Here I change "1/5" to "1/4" in original question.

<img width="638" height="1363" alt="image" src="https://github.com/user-attachments/assets/7dda237c-d03f-4099-90be-6ae0cba0fdbc" />

---

Here I change "one hat" to "two hat" to test its robustness.

<img width="645" height="1367" alt="image" src="https://github.com/user-attachments/assets/3f32fc17-4c49-4afe-bc15-3b8e5c523425" />

---

Another question where it's able to give the correct answer after multiple "self-rollouts":

<img width="962" height="1369" alt="image" src="https://github.com/user-attachments/assets/49a7e2e5-3035-4fef-ae1a-5aac3fbd4d22" />

---

Then I saw a paper (https://arxiv.org/abs/2507.14417) mentioning "misleading questions":

<img width="1305" height="1135" alt="image" src="https://github.com/user-attachments/assets/dbced9f7-4903-4651-acef-85b67df420d4" />

Let's test one.

<img width="1075" height="1162" alt="image" src="https://github.com/user-attachments/assets/b9aa7fca-acca-483d-9425-82c0ab3613a6" />

Another.

<img width="1066" height="1204" alt="image" src="https://github.com/user-attachments/assets/af0e5053-653a-4a07-b75f-a651255baf0c" />

Change a number.

<img width="1074" height="1164" alt="image" src="https://github.com/user-attachments/assets/4065b11f-05c2-43f0-8247-66f0c4fe687e" />

---

Some simple code question.

<img width="643" height="1372" alt="image" src="https://github.com/user-attachments/assets/b7552380-3a07-43cd-94d3-45c6e191b552" />

---

<img width="962" height="1364" alt="image" src="https://github.com/user-attachments/assets/1c63b596-e036-4fb7-830c-3c9798ee5c7a" />

---

Some evals:

<img width="1605" height="176" alt="image" src="https://github.com/user-attachments/assets/bcf95a8e-cb03-46dd-ac95-a33c4b18d84f" />

and https://huggingface.co/spaces/Jellyfish042/UncheatableEval :

<img width="1585" height="880" alt="image" src="https://github.com/user-attachments/assets/1e37b736-eae0-4b83-9d16-b4c2233088d9" />

Overall I think it's decent for pure RNN + pretrain 2T tokens :)

Future plan:

1. scale to 8T tokens (my current full datset) and beyond

2. add DeepEmbed (https://x.com/BlinkDL_AI/status/1926941496684519805) and maybe DeepEmbedAttention (https://x.com/BlinkDL_AI/status/1939532655823040545)

3. have been training RWKV-7 "G0" 13.3B too

---

UPDATE: It can solve some easy AIME too.

<img width="963" height="1373" alt="image" src="https://github.com/user-attachments/assets/fb815747-890b-427c-93c4-69d3ee54fc5c" />

<img width="962" height="1374" alt="image" src="https://github.com/user-attachments/assets/08460b8e-e499-4dca-ace4-5956cbdcd7a6" />

<img width="641" height="1373" alt="dce19e0e02cf5edff3ef5042ba3061a1" src="https://github.com/user-attachments/assets/96d0f1ae-9915-42ba-a60f-65e1d96140d6" />

Modifed question:
<img width="964" height="1370" alt="image" src="https://github.com/user-attachments/assets/c84e73c1-0dbf-4e67-a103-7cb990d743a0" />
