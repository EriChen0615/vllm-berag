# NarrativeQA Experiments

# Introduction

We should test the throughput of standard RAG and BERAG on a realistic dataset. 
    For BERAG to be useful, the task should 
        (1) have long contexts for response generation; 
        (2) ideally the context can be naturally partitioned into chunks for parallel processing; 
        (3) only one chunk is critically relevant for response generation; and 
        (4) have a training set for BERAG fine-tuning.  

NarrativeQA [1] is a dataset that meets all of the above requires. 
    Notably, it is adapted in LongBench [2], a widely-used long-context generation benchmark, as a single-document long-context QA task.
    NarativeQA has 7.81K document chunks of length from 100-1000 characters. 
        Each question is based on a specific chunk, and the ID of that relevant chunk is annotated in the dataset. 
        This construction makes it easy to construct long context while ensuring relevant information is included. 
        In LongBench, the dataset is adapted so that the average input length is 18,409 tokens, the longest in its Single-Document QA test suite.

We also want to establish reporting schemes using experiments on Narative QA.
    Importantly, our current BERAG implementation is based on offline, synchronous generation. 
        This can be extended in future work, which will make it possible to report serving metrics such as latency versus request rate as done in the PagedAttention paper [3].
    We will report the usual inference performance metrics: 90-percentile Time-To-First-Token (P90 TTFT), and Time-Per-Output-Token (P90 TPOT). 
    We will also report the end-to-end processing wall-time and request-per-second. 

This document is structured as follows:
    First, we will discuss how to adapt the Narrative QA dataset to form varying input lengths for testing, as well as quality evaluation metrics. 
    Then, we will define the inference performance metrics.
    We then proceed to report our experimental results. 

# Adapting Narrative QA

The Narrative QA data can be obtained via HuggingFace at https://huggingface.co/datasets/illuin-conteb/narrative-qa. There are two subsets: `documents` and `queries`.
    The `documents` subset contains 7.81k chunk-id-to-text pairs. 
        Notably, each document is chunked into smaller pieces, with the last suffix (e.g., `_2`) indicating the chunk number of a particular document. 
        Documents are shared between the training and test splits.
        Each chunk has an average of 150 tokens. 
    For `queries` subset contains 8.58k rows in its test split. 
        Each row consists of a query (we will use the `og_query` field, instead of `query`), a ground-truth chunk-id that corresponds to the relevant chunk for answering the query, and a gold answer.  
    Most of the gold answers are short, that is, under 50 tokens. 
        There are however longer response up to 224 tokens. 

We will generate different test sets from Narative QA with varying input lengths. 
    These lengths are controlled by the number of input chunks provided to the model. 
    For example, we can form a test set with K=10 chunks provided as input. 
        We make sure that the ground-truth chunk and the other chunks in the same document exists. 
        After incorporating these chunks, we will randomly include other chunks in the `documents` subset, so that the final count equals the specified K. 
    For our experiments we will create subsets of K=50, 75, 100, 150, 200.
        Note that these sets are created by a script and then stored statically. 
        These should corresponds to input length with average 30K tokens.
    In storing the dataset, we will store the document chunks as a list. 
        This will allow us to use the same static dataset for standard RAG and BERAG inference. 

We will use BLEU (the `sacrebleu` implementation) to evaluate the quality of model generation against the reference gold answer. 

# Inference Performance Metrics

In this section, we define the metrics for evaluating inference performance.

`P90/P50 TTFT`
    This refers to 90- or 50-percentile Time-To-First-Token.
    In vLLM, TTFT is reported as:
        `TTFT = iteration_timestamp - req_stats.arrival_time`
        which includes waiting time. 
        TTFT can be further decomposed into `queued_time` and `prefill_time`
            `queued_time` refers to the duration between admitting the request to the scheduler acutually schedules the request to send to the worker.
                That is, `queued_time = scheduled_ts - queued_ts`
            `prefill_time` refers to the duration between scheduled time and first token output time. It is the duration for the model to process the request and generate a token.
                That is `prefill_time = first_token_ts - scheduled_ts`
            As we are doing offline, synchronous generation, the `queued_time` is likely to be uninformative. Therefore, we will also report `P90/P50 Prefill Time`. 
    For BERAG, TTFT the time when the first sampling is done, that is, when all children branches finish their first pass and a token is returned from the worker. 
        Each request should log its TTFT, i.e., when the first token is returned. 
            Note: vLLM can attach per-request timing states to each RequestOutput. For offline, synchronous generation, this is turned off by default. We need to check if the stats report genuine TTFT time in offline, synchronous calls. 
                We need to make sure this is logged correctly.
            Similarly, we also want to separate `queued_time` and `prefill_time`. We should use the parent request to carry these information as a group.
                Importantly, the `queued_ts` (queued timestamp) should record when the parent request is expanded into K children request. 
            The `scheduled_ts` should record when the FIRST shard of the BERAG group is scheduled for the model. 
                This is because if there are a large number of shards to process, the model may take multiple forward passes (i.e., the scheduler may distribute shards of the same BERAG group to several iterations). All iterations should count. 
            Because each child request is a standard vLLM request, the parent request should be able to compute the group's overall `queued_ts` and `scheduled_ts` from the individual record. In particular:
                `queued_ts = min([child.queued_ts for child in children])`
                `scheduled_ts = min([child.scheduled_ts for child in children])`
        Similar to standard RAG, we will also report `P90/P50 Prefill Time`
    

`P90/P50 TPOT`
    This refers to 90- or 50-percentile Time-Per-Output-Token. 
    For standard vLLM request:
        We should make sure these are correctly reported in our offline, synchronous setup. 
        This is computed as `(last_token_ts - first_token_ts) / (# tokens - 1)`. 
    For BERAG:
        we should do something similar, but make sure these are computed for at the parent request. Though the children requests should have the same statistics as they share output tokens. 

As a general guide, we will make the parent request TTFT and TPOT directly readable, instead of logging individual group metrics. 

`Total Wall Time`
    This is the total wall time between when the engine is initalized and ready and completing all requests. 

`Requests per second`
    This is simply the number of requests divided by the total wall time. 

# Experimental Setup

Model
    We will use the [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) model. 
    The model supports context length up to 262,144 natively.

Data
    We will form K=50, 75, 100, 150, 200 using Narrative QA data as described above. We will take a subset of 2048 examples, which are issued by one synchronous vLLM call. 
    These will be static datasets, saved under `my_outputs/data/NarrativeQA`


Standard RAG
    For standard RAG, we will use normal vLLM requests for generation. 
    Write a dedicated script for benchmarking standard RAG that iterates through each K subset. 
        It should be easy to set which K-set to run, potentially by declaring a list to iterate over

BERAG
    We will use the BERAG vLLM requests for generation
    We need to make sure that the metrics are correctly returned before running the final experiments. 
    Write a dedicated script for BERAG inference, similar to that for standard RAG.

Evaluation
    We will report the inference metrics as well as the BLEU score w.r.t the gold answer. 
    The output texts should be saved under `my_outputs/experiments/<exp_name>`, along with the metrics.
    We will also report mean input/output tokens, assuming that these are available from vLLM request stats.

Prompt
    We will use the LongBench prompt from https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json. 
        We will place a generic system prompt: "You are a helpful assistant", and place the long bench propmt in user message. 
    We will apply the chat templates of Qwen ourselves and use the `llm.generate()` API in vLLM. 
        This will make sure BERAG and RAG sees the same inputs. 
 


    





# References

[1] The NarrativeQA Reading Comprehension Challenge, ACL 2017, https://arxiv.org/abs/1712.07040 
[2] LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding, ACL 2024, https://arxiv.org/abs/2308.14508
[3] Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023, https://dl.acm.org/doi/10.1145/3600006.3613165 
