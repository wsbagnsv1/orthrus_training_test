This code is an probably working implementation of the orthrus papers trainingcode. 
It is not yet fully validated at scale and might not be fully optimized but overall it should mostly work. 
I trained a the 130m smollm2 on the smoltalk dataset and dtrained it at effective batch size 64 to 6000 steps and got this performance out of it:

<img width="1229" height="603" alt="Capture" src="https://github.com/user-attachments/assets/9495f8d8-c0bc-4ad7-b136-edc8850db5fd" />

With more training, bigger model and better dataset for example the one the othrus guys used in their paper i expect this to speed stuff up much better. Also the inference can be optimized as well which would help too ofc.

Thanks to the guys behind orthrus for their paper [Memory-Efficient Parallel Token Generation via Dual-View Diffusion](https://arxiv.org/abs/2605.12825)!

Their github repo: 
https://github.com/chiennv2000/orthrus

(triton kernels are currently NOT better than flex attention and currently not working but might be interesting for future optimization.

I have added qwen3.5 training into a seperate branch, that one is full of trash and not optimized I intend to clean it up later, since i used gemini to cook up some custom kernels and that was creating a whole lot of test files, but in the end it seems to work decently. The trainig itself is better than pure pytorch training with those kernels but they are far from perfect. Im realtively sure the math behind it works but im not perfect so if you find errors please let me know. Also obviously I intend to clean the codebase up once i have decently optimized and validated it with real world testing (; 

Inference of a qwen3.5 0.8b orthrus trained for 40ish steps with the trainings code:
<img width="1900" height="809" alt="Capture" src="https://github.com/user-attachments/assets/f691ce51-fd67-477c-ab37-f01789647c6a" />
