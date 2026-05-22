This code is an working implementation of the orthrus papers trainingcode. 
It is not yet fully validated at scale and might not be fully optimized but overall it should mostly work. 
I trained a the 130m smollm2 on the smoltalk dataset and dtrained it at effective batch size 64 to 6000 steps and got this performance out of it:

<img width="1229" height="603" alt="Capture" src="https://github.com/user-attachments/assets/9495f8d8-c0bc-4ad7-b136-edc8850db5fd" />

With more training, bigger model and better dataset for example the one the othrus guys used in their paper i expect this to speed stuff up much better. Also the inference can be optimized as well which would help too ofc.

Thanks to the guys behind orthrus for their paper [Memory-Efficient Parallel Token Generation via Dual-View Diffusion](https://arxiv.org/abs/2605.12825)!

Their github repo: 
https://github.com/chiennv2000/orthrus

(triton kernels are currently NOT better than flex attention and currently not working but might be interesting for future optimization.
