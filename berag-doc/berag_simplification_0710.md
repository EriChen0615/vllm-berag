# BERAG Simplication and MM Inputs 0710 

# Introduction

We'd like to simplify the BERAG implementation for now, and prioritize MMInput Handling.

The core objective is to be able to run BERAG inference with vLLM on E-VQA. Notably, E-VQA has a query image but no passage images. This means that every BERAG child request should share the query image's multi-modal encoding. We should make sure NOT to repeatedly encode it. 

There are also a few simplifying assumptions we can make for this initial version of BERAG vLLM:
* We can assume that the scheduler only sends *complete* BERAG groups. 
    * That is, every iteration would contain N complete BERAG gruops. Let K denotes the number of documents per BERAG request and N_max the maximal batch size, we have K*N <= max. We choose the maximal K such that this remains true. 
    * This allows us to leave the partial forward case later (i.e., where only a subset of BERAG children receive forward computation). This simplifies prior, logits caching, etc.
* We will also make sure that in the first pass, we generate the first token right away. Importantly, we shouldn't need the GPU worker to report prior ready and wait for another iteration for generation. This is because we have made the simplifying assumption above. 
* We will make sure that scheduler works at the BERAG GROUP level, not the requests level and then rebundle them into BERAG groups. 
