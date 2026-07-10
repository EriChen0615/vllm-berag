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
    Specifically, consider a common prefix of length A, K documents each having D tokens. The prefill compute required by BERAG is $O(A^2) + O(K\times D^2)$ (assuming prefix reuse), where as for standard RAG the prefill compute is $O((A+K \times D)^2)$. 
    In auto-regressive decoding, BERAG can be faster than standard RAG by dropping low-posterior document branches. 

However, it is not straight-forward to implement BERAG inference that is as fast as optimized standard RAG. 
    Modern inference engines like vLLM handles generic sequence completions, and does not implement efficient scheme for parallel, ensemble-based generation. 
    Notably, in BERAG, each branch in the ensemble requires access to shared KV cache for the common prefix and individual KV cache for its document singleton. This is not immediately available in engines like vLLM. 
    There are a few other minor technical discrepancies: 
        BERAG requires an additional MLP layer that consumes a last-layer embedding for computing the prior distribution, which would require careful tracking of hidden states. 
        BERAG aggregates the distributions from all branches at each decoding step. This again requires tracking the related sequences carefully and making sure that all branches have produced their current-step logits before sampling the next token. 

Despite these challenges, the throughput optimization techniques such as PagedAttention[2] is conceptually situable for implementing BERAG inference. 
    In fact, as one of the motivations for PagedAttention[2], the authors note: *"LLM services often use advanced decoding algorithms, such as parallel sampling and beamsearch, that generate multiple outputs per request. In thesescenarios, the request consists of multiple sequences that can partially share their KV cache. However, memory sharing is not possible in the existing systems because the KV cache ofthe sequences is stored in separate contiguous spaces."*
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

In the following description, we follow the life-cycle of a user request through the system. 
    To avoid complication from online serving and frontend-backend communication, we focus on synchronous, offline generation. 

### Life-Cycle of a User Request

A user request (i.e., user prompt) is sequentially transformed into the following objects as it is being processed:

The overall process is

```
User prompt
-> EngineCoreRequest
-> Scheduler Request
-> repeated:
       SchedulerOutput
       -> GPU worker request state
       -> model forward / sampling
       -> ModelRunnerOutput
       -> Scheduler.update_from_output
       -> Scheduler Request updated
       -> EngineCoreOutput emitted when there is frontend-visible output
-> OutputProcessor
-> RequestOutput
```

We now describe each phase and object in detail. 


-----Admission Phase Starts-----

**User prompt**
    [Endpoint API Request]
    may be string, token IDs, multimodal input, plus Sampling Params

-> **EngineCoreRequest**
    [Translated Request for the Engine]
    raw prompt -> tokenized input
    arrival time, priority, LoRA, multi-modal features -> attached
    request_id -> assigned

-> **Scheduler Request**
    [In-Engine Representation of a Request]
    Added status (WAITING or RUNNING)
    initialize `output_token_ids`, `block_hashes`, `preemption counters`, etc.

-----Admission Phase Ends------

-----Iteration Loop Starts------

-> **SchedulerOutput**
    [The Scheduler's per-step execution plan]
    dictates:
        which requests run in this iteration
        how many tokens each request should compute
        whether this is prefill or decode
        which KV blocks each request should use
            to evict/preempt a sequence, the scheduler sends `finished_req_ids` and `preempted_req_ids` to the worker. The worker than removes the local state for that request. 
        which requests are already cached on the worker
        which requests finished or were preempted

-> **GPU worker request state**
    [The GPU local tensor-formatted data that corresponds to active reqeusts]
    request_id -> worker-local request index
    prompt/token data -> GPU/CPU token buffers
    KV block IDs -> block tables
        Note that the The scheduler owns authoritative KV block allocation and sends block IDs through `SchedulerOutput`. The worker mirrors those IDs into execution-local BlockTables used by attention kernels.
    scheduled tokens -> input batch
    attention info -> attention metadata
    sampling params -> sampler state

    The worker performs:
        InputBatch + attention metadata + block tables
        -> model forward
        -> hidden states
        -> logits
        -> sampler

-> **ModelRunnerOutput**
    [Worker's result for one row/request in the batch]
    It is a batch object with per-request rows indexed by `req_id_to_index`. 
        A row may have no sampled token during non-final prefill chunks. 
    produced:
        sampled_token_ids
        logprobs (if requested)
        KV connector output
            worker's reply about asynchronous KV-cache transfer state, so that the scheduler can keep its KV ownership and request states consistent.
    
    The scheduler will consume this and updates its internal request states. 

-> **Scheduler.update_from_output(...)**
    sampled token ids -> append to scheduler Request
    stop conditions -> checked
    finished requests -> freed
    KV ownership -> released if done
    EngineCoreOutput -> created

-----Iteration Loop Ends-----

-----Frontend Output Phase Starts-----

-> **EngineCoreOutput**
    [A scheduler-authored output delta for the frontend]
    carries:
        request_id
        new_token_ids
        finish_reason
        stop_reason
        logprobs
        events / stats

-> **RequestOutput**
    [User-facing output object]
    token ids -> detokenized text
    ......
    then `LLM.generate()` eventually returns. 

-----Frontend Output Phase Ends-----

### Beam Search Implementation

Beam search is also a sequence-level decoding scheme that keeps multiple candidates. Its implementation may be instructive for BERAG.

The current beam search implementation in vLLM is a wrapper above normal generation requests. 
    Each active beam is turned into a normal, one-token request. 
    vLLM runs those requests, return log-probabilities, and the python beam search loop expands/prunes beams. 

Unfortunately, this is problematic for BERAG implementation. An efficient BERAG implementation likely needs to incur changes in the scheduler and model runner because:
    * BERAG requires the scheduler to know about how each branch belongs to an overall request.
    * Gang scheduling is required. At each decode step, BERAG needs to schedule all active branches for a request in order to advance the generation.
    * Branch dropping affects KV freeing. 

## BERAG Generation

We now review the generation process of BERAG. Given input x and K documents z1,...,zk, BERAG forms K branches (x,z1), (x,z2), ..., (x,zk). 

In the first pass, the model generates a prior distribution m1,...,mk over the K documents. 
    This distribution is generated by using the last-layer embedding of the fourth latest token in the input to generate a logit. Then the logits of all branches are passed through a softmax operation to give a distribution.

Let p1(t),...,pk(t) denotes the output vectors for the next-token distributions of branches 1,...,k at time step t. Each vector has dimension |V|, where |V| is the vocabulary size. Let q1(t),...,qk(t) denotes the posterior distribution. We have:

p(t) = \sum_k pk(t)qk(t)
qk(t) = qk(t-1)pk(t-1, y(t-1)) / (\sum_k qi(t-1)pk(t-1, y(t-1)))

where y(t-1) denotes the generated token at step t-1. y(t) is sampled from p(t) with greedy sampling. 

At the first step, the document posterior qk(1) is the document prior mk. We assume p(0, "") = 1 for boundary condition (i.e., the probability of the empty string "" before generation is 1).

### BERAG Inference Optimization

We now describe the optimizations required to implement BERAG efficiently. 

Shared Prefix Encoding:
    The shared input context x should be encoded only once and shared for all subsequent branches.
        This is especially important in multi-modal contexts, where encoding images can be costly.

Top-P Pruning:
    Branches with small posterior weight should be dropped, such that the cumulative probability of the remaining branches exceed a predefined value (e.g., 1 - 1/2K as in [1]). 
        This can drastically speed up generation by reducing the effective context.

Gang Scheduling:
    Branches that correspond to a request should ideally be processed in one batch. Note that a request cannot advance before all active branches receive a forward pass.

# Implementation

We now describe how to incorporate BERAG into vLLM. We build our work on the following basis and assumptions:
    * vLLM v1 architectures.
    * offline, synchronous generation.
    * single GPU. 
    * text-only.
    * K child requests per BERAG request.
    * Make use of the Automatic Prefix Caching (APC) mechanism already implemented by vLLM.
    * No common prefix encoding yet. So prefill compute is O(K * (A+D)^2), though we will consider shared multimodal input later. 

## Overview of Changes

We introduce a BERAG request mode, `BERAGUserRequest`, which expands into K internal vLLM child requests, one per retrieved document branch.
    These child requests are ordinary vLLM requests for prefill/KV purposes, but they carry BERAG metadata that ties them to one group.

The scheduler maintains a `BeragGroupState` for each BERAG user request.
    This state owns the group lifecycle, active/pruned branch set, prior and posterior weights, and the logical BERAG decode step. The scheduler remains the authority for KV allocation;

A key distinction is that BERAG branch forwards may be scheduled at branch-shard granularity, while token generation is committed at group-step granularity. 
    A single BERAG decode step may span multiple physical model-runner batches. 
    The shared token is sampled only after all active branches in the group have contributed evidence for that logical step.

Overall, a `BERAGUserRequest` undergoes the following steps:

```text
Admission:
    BERAG user request -> K child EngineCoreRequests
    scheduler registers a BERAG group, tying these children together

Prefill:
    children prefill [shared_prefix][document_k][suffix]
        shared_prefix reuse relies on APC
    prior hidden states are extracted at the configured prior positions
    prior scores are collected until all active branches have priors

Group-synchronous decode loop:
    scheduler schedules branch shards from one or more BERAG groups
    model runner computes branch logits and next-token likelihood from the branches
    partial evidence is accumulated by group and logical decode step
    once all active branches in a group have contributed:
        BERAG sampler samples one shared token
        posterior is updated
        scheduler appends the shared token to all active child Requests
        branches may be pruned and their KV state released
    loop continues until group-level stop
```

## BERAG Decode Executation

To fully utilize GPU memory, BERAG introduces a distinction between a physical model-runner batch and a logical BERAG decode step.
    A *physical batch* is one normal vLLM scheduler/model-runner iteration. It may contain ordinary requests and BERAG branch shards from one or more BERAG groups.
    A *logical BERAG decode step* is complete only after every active branch in the group has contributed its next-token evidence for the same step_id.

The scheduler may therefore schedule:

    batch 1: group A branches 0..49, group B branches 0..29
    batch 2: group B branches 30..49, group C branches 0..59

Group A can commit its shared token after batch 1. Group B cannot commit until batch 2 completes.

Most BERAG shard outputs do not contain sampled token IDs. They only mark that some branches have contributed evidence. The final shard for a group step triggers grouped sampling, posterior update, and token commit.

We will illustrate this with numbered operation sequences below:

```markdown
1. Scheduler selects BERAG branch shards.
2. Scheduler emits ScheduledBeragShard in SchedulerOutput.
3. Model runner computes logits for scheduled branch rows.
4. Model runner accumulates group evidence by group_id and step_id.
5. If shard is not final, return partial BeragModelRunnerOutput.
6. If shard completes the group:
       sample shared token
       update posterior
       return sampled_token_id and new_log_posterior
7. Scheduler appends sampled token to all active child Requests.
8. Scheduler advances group step_id.
```

We declare our design choices explicitly below:

```markdown
1. The model runner owns BERAG accumulators.

Because branch logits are GPU tensors, full-vocabulary evidence should stay on GPU. The scheduler only tracks compact CPU-side state such as pending branches, completed branches, priors, posteriors, and step IDs. In the first design we assume a single GPU, so all shards for a BERAG group naturally run on the same model runner.

2. Evidence accumulation is exact.

For each branch shard, the model runner computes the full-vocabulary branch distribution and accumulates:

    p_mix(v) += q_k * p_k(v)

where q_k is the current posterior weight for branch k. The accumulator remains on GPU until the logical BERAG step is complete. Sparse Top-K evidence is a possible future optimization, but it changes the sampling distribution and is not part of the base design.

3. The scheduler decides whether a shard is final.

The scheduler owns `pending_branch_ids` and `completed_branch_ids` for each BERAG group and step. When it schedules a shard, it can determine whether that shard completes the logical step. It marks this in `ScheduledBeragShard`.
```

## Scheduler Policy

Groups are scheduled in a First-Come-First-Served (FCFS) basis. 
    The oldest in-progress BERAG group will be scheduled first. 
    If any capacity remains, the next in-progress group will be schedueld.
    If no in-progress groups remain, start the oldest waiting group. 

Each physical batch has the following priority with respect to scheduling:

```markdown
1. Select the oldest BERAG group with an in-progress logical decode step.
2. Form a shard from as many of that group's pending branches as fit.
3. If capacity remains, form shards for the next oldest in-progress groups.
4. If no in-progress group has pending branches, start the oldest waiting group
   by forming its first shard.
5. Continue forming shards for later waiting groups until the physical batch is full.
```

The scheduler does not schedule an entire BERAG group unless it fits. It schedules BeragShards: subsets of pending branches from a group for a particular step_id.

## BERAG Accumulator

The BERAG accumulator lives on the GPU worker. It accumulates next-token probabilities from branch shards before producing the group-level mixture distribution.

The GPU worker preallocates one fixed accumulator workspace before vLLM's KV-cache memory profiling:

```text
berag_workspace: [num_accumulator_rows, vocab_size]
```

The workspace is treated as a row pool. The worker owns the tensor storage. The scheduler owns a small page-table-style row allocator and decides which rows belong to each `(group_id, step_id)` and branch. These row IDs are carried in `ScheduledBeragShard`; the worker only reads and writes the assigned rows.

Rows are assigned dynamically:
    * one row stores the mixture distribution for an active `(group_id, step_id)`.
    * one row stores branch evidence for an active branch that has contributed to that step.

The worker reports workspace capacity and row usage back to the scheduler. The scheduler uses this to avoid launching shards that need unavailable rows. Row release is scheduler-driven: when a branch is pruned, its branch row is released; when a group step commits, all rows for that `(group_id, step_id)` are released.

Therefore capacity is controlled by the total number of live accumulator rows, not by `max_active_berag_groups * max_branches_per_group`. If many documents have been dropped, their rows are not reserved.

The memory cost is:

```text
num_accumulator_rows * vocab_size * bytes_per_value
```

For `vocab_size=152k` and bf16, one row is about 0.29 MB. A 200-row workspace is about 58 MB; a 400-row workspace is about 116 MB.
    For reference, a 400-row workspace would accommodate 8 BERAG groups with K=50 simulataneously.

If the row pool is exhausted, the scheduler should not launch shards that require new accumulator rows. It can continue shards whose rows already exist, or wait until a group step commits and releases rows.

# References

[1] BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering, arxiv 2026, https://arxiv.org/abs/2604.22678
[2] Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023, https://dl.acm.org/doi/10.1145/3600006.3613165
[3] Orca: A Distributed Serving System for Transformer-Based Generative Models, OSDI 2022, https://www.usenix.org/system/files/osdi22-yu.pdf 

# Appendix


## Data Model

We now describe the necessary new data models introduced to implement the above procedure.

`BeragChildMetadata`
    * Lives on both `EngineCoreRequest` and the scheduler `Request`.
    * Identifies a normal-looking vLLM request as one branch of a BERAG group.
    * Should stay small and mostly immutable.
    * Minimal fields:
        * `group_id`
        * `branch_id`
        * `prior_token_index`
    * Group-level facts such as `parent_request_id`, `num_branches`, posterior weights, and phase belong to `BeragGroupState`, not to each child request.

`BeragGroupState`
    * Owned by the scheduler.
    * Authoritative state for one BERAG user request.
    * Tracks group lifecycle and branch state:
        * `group_id`
        * `parent_request_id`
        * `child_request_ids`
        * `active_request_ids`
        * `branch_ids`
        * `phase`
        * current logical decode `step_id`
        * `pending_branch_ids` and `completed_branch_ids` for the current step
        * `prior_scores`
        * `log_prior`
        * `log_posterior`
        * `top_p_pruning`
    * Should not store GPU tensors or full-vocabulary distributions. Large logits/probabilities should be consumed in the model runner or kept in a GPU-side accumulator for staged decoding.

`ScheduledBeragShard`
    * Carried inside `SchedulerOutput`.
    * Describes the subset of BERAG branches scheduled in the current physical model-runner batch.
    * Replaces the assumption that an entire BERAG group must fit in one batch.
    * Tells the model runner:
        * `group_id`
        * logical `step_id`
        * scheduled `request_ids`
        * corresponding `branch_ids`
        * current posterior weights for those branches
        * prior-token positions for branches that still need prior extraction
        * whether this shard completes the group step, if known
    * The model runner resolves packed row indices itself because it owns `InputBatch`, `logits_indices`, and the flattened hidden-state layout.

`BeragModelRunnerOutput`
    * Carried back inside `ModelRunnerOutput`.
    * Returns compact BERAG-related updates from the model runner to the scheduler.
    * May represent either:
        * a partial shard output, where branch evidence has been collected but no shared token is sampled yet; or
        * a final group-step output, where a shared token and posterior update are available.
    * Tracks:
        * `group_id`
        * `step_id`
        * completed `branch_ids`
        * `prior_scores`, if newly available
        * `sampled_token_id`, only when the logical group step completes
        * `new_log_posterior`, only when available
        * `pruned_branch_ids`
