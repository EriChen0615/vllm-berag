# Implementing BERAG in vLLM for high-throughput generation

# Introduction

Bayesian Ensemble Retrieval Augmented Generation (BERAG)[1] is an inference procedure that forms a weighted ensemble of $K$ branches during generation. 
    Each branch is conditioned on a document singleton. 
    The next-token distribution from each branch is weighted to form the final, overall next-token distribution to sample from at each decoding step. 
    The weights of these branches are updated following Baye's rule during generation, that is, an initial prior distribution over the relevance of the branches gets iteratively refined as more tokens are generated. 

BERAG offers several attractive features over standard RAG that conditions on a single, long context from concatenating all documents in generation. 
    * Memory Parallelism. BERAG could schedule forward passes of each branches sequentially. This makes it possible to condition on deep retrieval list (e.g., with Top-50 documents) under reasonable VRAM budget.
    * Positional Invariant. BERAG forms an ensemble that is invariant to how the input documents are ordered. This eliminates the ``lost-in-the-middle'' effect. 
    * Interpretable Evidence Selection. The prior and posterior distributions over document singletons can be interpreted as how much a particular singleton contributes to the overall generation. This can be useful for interpretability, and also for speeding-up inference as branches with low weights can be dropped during generation. 

BERAG should also have competitive inference efficiency compared to standard RAG.
    Specifically, consider a common prefix of length A, K documents each having D tokens. The prefill compute required by BERAG is $O(A) + O(K\times D)$, where as for standard RAG the prefill compute is $O((A+K \times D)^2)$. 
    In auto-regressive decoding, BERAG can be faster than standard RAG by dropping low-posterior document branches. 

However, it is not straight-forward to implement BERAG inference that is as fast as optimized standard RAG. 
    Modern inference engines like vLLM handles generic sequence completions, and does not implement efficient scheme for parallel, ensemble-based generation. 
    Notably, in BERAG, each branch in the ensemble requires access to shared KV cache for the common prefix and individual KV cache for its document singleton. This is not immediately available in engines like vLLM. 
    There are a few other minor technical discrepancies: 
        BERAG requires an additional MLP layer that consumes a last-layer embedding for computing the prior distribution, which would require careful tracking of hidden states. 
        BERAG aggregates the distributions from all branches at each decoding step. This again requires tracking the related sequences carefully and making sure that all branches have completed before sampling the next token. 

Despite these challenges, the throughput optimization techniques such as PagedAttention[2] is conceptually situable for implementing BERAG inference. 
    In fact, as one of the motivations for PagedAttention[2], the authors note: *"LLM services often use advanced decoding algorithms, such as parallel sampling and beamsearch, that generate multiple outputs per request. In thesescenarios, the request consists of multiple sequences that canpartially share their KV cache. However, memory sharing isnot possible in the existing systems because the KV cache ofthe sequences is stored in separate contiguous spaces."*
    In implementing beam search, they note that *"multiple sequences with one request (as in beam search) are gang-scheduled as a sequence group. They are always preempted or rescheduled together."*
    These mechanism should make an efficient BERAG implementation possible.

This project aims to integrate efficient BERAG inference in vLLM as a proof-of-concept. The goal is to show:
    * BERAG inference can be implemented into modern inference engines like vLLM. 
    * BERAG inference realizes the promise of its theoretical compute advantage compared to standard RAG, given similar optimization like PagedAttention is applied. 
    * Understand the key modifications required to implement BERAG inference efficiently in an established inference engine, as well as the limitations. 

This document will proceed as follows:
    * Background
        * First, we review the relevant optimization techniques and architecture of vLLM in implementing efficient transformer inference. We will focus on the case of a single GPU, and leave distributed, multi-GPU inference for future work. 
        * Then, we review the BERAG inference procedure, and highlight the necessary high-level changes required to implement BERAG efficiently. This completes the technical background section.
    * Implementation
        * In this section, we describe the specific implementation of efficient BERAG inference under the vLLM architecture.
    * Results
        * In this section, we report inference performance for BERAG using our variant of the vLLM engine and compared its performance to standard RAG using the unmodified vLLM engine. 

# Background

## KV Cache

In Transformer-based model, the Key and Value vectors of an attention block is used at every decoding step. These vectors can be cached to avoid expensive recomputation. The cached vectors are known as *KV Cache*. 
    Modern inference engine provides implementation that manges KV cache efficiently. That is, they implement swapping and eviction schemes to maximize reuse of computed KV Cache. 
    This usually requires careful scheduling at the iteration-level (i.e., at the level of each decoding step) and implementing custom GPU kernel for efficient memory movements. 
        For example, a custom kernel may fuse reshape and block write operations in a single operation to reduce the number of launches of a GPU device. 

## Paged Attention

Paged Attention [2] introduces a level of indirection for memory access, as done in the context of Operating System. 
    KV cache appears to be stored in *logical* KV blocks that are continuguous to the inference process. However, they may reside in discontinous *physical* KV blocks that reside at different locations of the GPU memory. 
    The translation between logical and physical addresses is performed by a *page table* (or *block table*) as done in OS. Like in OS, this form of virtual memory avoids memory fragmentation and maximizes memory utilization. 
    With Paged Attention, the vLLM engine can run inference at much larger batch sizes and efficiently reused common prefix KV Cache. 
    In OS, memory is manged in pages of fixed sizes. In Paged Attention, KV cache is manged in blocks of fixed sizes, usually with 4-16 tokens worth of GPU memory per KV block. 

> N.B., For BERAG, Paged Attention provides a natural mechanism to re-use prefix KV cache. 

## Iteration-level Batching

Iteration-level batching is proposed to address deficiencies in request-level batching.
    For request-level batching, the serving engine becomes free only after all requests (i.e., sequence to complete) are completed. 
        However, many sequences would have terminated early. Such design prevents the model from returning these early finishers as soon as they are ready. 
    Iteration-level batching advances requests one-token-at-a-time. 
        The scheduler receives execution result on every iteration (i.e., decoding step), making it possible to detect early-exit and fully utilize the compute resources. 
    
> N.B., BERAG naturally operates at the interation level. 

## The vLLM architecture




# References

[1] BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering, arxiv 2026, https://arxiv.org/abs/2604.22678
[2] Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023, https://dl.acm.org/doi/10.1145/3600006.3613165
[3] Orca: A Distributed Serving System for Transformer-Based Generative Models, OSDI 2022, https://www.usenix.org/system/files/osdi22-yu.pdf 

