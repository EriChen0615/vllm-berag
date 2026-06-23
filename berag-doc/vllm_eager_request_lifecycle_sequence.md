# vLLM Offline Eager Request Lifecycle

This file uses conservative Mermaid sequence-diagram syntax for compatibility
with older VS Code Mermaid preview extensions.

## Current vLLM Request Lifecycle

```mermaid
sequenceDiagram
    participant Caller
    participant LLM
    participant OfflineInferenceMixin
    participant LLMEngine
    participant InputProcessor
    participant OutputProcessor
    participant EngineCoreClient
    participant EngineCore
    participant Request
    participant Scheduler
    participant ModelExecutor
    participant GPUModelRunner
    participant Sampler

    Caller->>LLM: generate
    LLM->>OfflineInferenceMixin: run completion

    OfflineInferenceMixin->>LLMEngine: add request
    LLMEngine->>InputProcessor: process inputs
    InputProcessor-->>LLMEngine: EngineCoreRequest
    LLMEngine->>OutputProcessor: add frontend request state
    LLMEngine->>EngineCoreClient: add request
    EngineCoreClient->>EngineCore: preprocess add request
    EngineCore->>Request: create scheduler request state
    Request-->>EngineCore: Request
    EngineCore->>Scheduler: add request
    Scheduler-->>EngineCore: queued request

    loop each engine step
        OfflineInferenceMixin->>LLMEngine: step
        LLMEngine->>EngineCoreClient: get output
        EngineCoreClient->>EngineCore: step

        EngineCore->>Scheduler: schedule
        Scheduler-->>EngineCore: SchedulerOutput batch

        EngineCore->>ModelExecutor: execute model
        ModelExecutor->>GPUModelRunner: execute model
        GPUModelRunner->>GPUModelRunner: update request state
        GPUModelRunner->>GPUModelRunner: prepare input batch
        GPUModelRunner->>GPUModelRunner: prepare attention metadata
        GPUModelRunner->>GPUModelRunner: run model forward
        GPUModelRunner-->>ModelExecutor: forward complete
        ModelExecutor-->>EngineCore: execute result

        EngineCore->>Scheduler: get grammar bitmask
        Scheduler-->>EngineCore: grammar output

        EngineCore->>ModelExecutor: sample tokens
        ModelExecutor->>GPUModelRunner: sample tokens
        GPUModelRunner->>GPUModelRunner: compute logits
        GPUModelRunner->>Sampler: sample
        Sampler-->>GPUModelRunner: sampled tokens
        GPUModelRunner-->>ModelExecutor: ModelRunnerOutput
        ModelExecutor-->>EngineCore: ModelRunnerOutput

        EngineCore->>Scheduler: update from output
        Scheduler->>Request: append sampled token ids
        Scheduler->>Request: check stop conditions
        Scheduler-->>EngineCore: EngineCoreOutputs

        EngineCore-->>EngineCoreClient: EngineCoreOutputs
        EngineCoreClient-->>LLMEngine: EngineCoreOutputs
        LLMEngine->>OutputProcessor: process outputs
        OutputProcessor->>OutputProcessor: detokenize
        OutputProcessor->>OutputProcessor: build RequestOutput
        OutputProcessor-->>LLMEngine: RequestOutputs
        LLMEngine-->>OfflineInferenceMixin: RequestOutputs
    end

    OfflineInferenceMixin-->>LLM: final sorted outputs
    LLM-->>Caller: request outputs
```

## BERAG Modification Pressure Points

```mermaid
sequenceDiagram
    participant BERAGRequest
    participant Scheduler
    participant RequestGroup
    participant PrefixKV
    participant BranchKV
    participant GPUModelRunner
    participant Aggregator

    BERAGRequest->>Scheduler: enqueue branch requests
    Scheduler->>RequestGroup: create branch group
    Scheduler->>PrefixKV: allocate shared prefix blocks
    Scheduler->>BranchKV: allocate branch document blocks
    Scheduler->>GPUModelRunner: schedule branch batch
    GPUModelRunner-->>Aggregator: branch logits and hidden states
    Aggregator->>Aggregator: combine branch distributions
    Aggregator->>Aggregator: sample shared next token
    Aggregator->>Aggregator: update posterior weights
    Aggregator-->>Scheduler: token and branch metadata
    Scheduler->>RequestGroup: append token to active branches
    Scheduler->>Scheduler: prepare next decoding step
```
