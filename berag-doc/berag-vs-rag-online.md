# BERAG versus RAG Online (Batch size = 1) Comparison

We'd like to compare the inference performance (TTFT, TPOT) of BERAG and RAG. The experimental setup is as follows:

Data: we will use the NarrativeQA set with K = 50, 100, 150, 200
* For the Multi-modal input, we will use a random image `my_data/just-a-random-picture.webp`. 
* We will take 256 samples for inference.

Inference: we will set batch size = 1. That is, vLLM will process 1 request at a time. 
* We will set the maximal output token to 32.
* Prompt: we will instruct the model to first describe the image, and then answer the question.
* We will keep to the standard hyper-parameters, e.g., max-num-seqs=256. Use max-model-len=40000. 
* Will do right-truncation, if exceeded maximal length. 

BERAG setting
* Use 512 accumulator rows

Metrics:
* We will compare the usual metrics (P50/90 TTFT, P50/P90 TPOT). 
* When the experiments complete, we should save a table like follows:
* In taking these numbers for a run, ignore the first row, as it may have included warm start time. 

Model:
* We will use Qwen3-VL-2B-Instruct for this experiment. 
* Note that the experiment directory should state the model slug to identify experiments. 

            K = 50     | K = 100   | K = 150 | K = 200
            TTFT, TPOT | TTFT, TPOT | ...
RAG     |
BERAG   |

Where the P50 numbers are reported. 