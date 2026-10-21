# SecondThought

Thinking before a tool call makes small models more accurate, but it costs six to seven times the output tokens and
makes them call tools they should leave alone. SecondThought thinks only when the direct call looks unsure.

Make the normal no-think tool call with log-probabilities on. If its mean token log-probability is low, rerun with
thinking. If the model declined to call, keep the decline. No training, no extra model: it uses numbers the server
already returns.

Paper: [`paper/secondthought.pdf`](paper/secondthought.pdf)

## Results

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) single-turn categories (1,398 items, including 240 where no
tool fits), Qwen3 on vLLM, accuracy over all items:

| | 1.7B | 4B | 8B |
|---|---|---|---|
| direct, no thinking | 0.803 | 0.850 | 0.858 |
| always think | 0.835 | 0.883 | 0.874 |
| **think on the least confident 30%** | **0.824** | **0.873** | **0.877** |
| random 30% | 0.818 | 0.857 | 0.870 |
| output tokens: always think | 317 | 354 | 364 |
| output tokens: gated 30% | 161 | 179 | 188 |

- The call's mean token log-probability predicts whether it's correct (AUROC 0.82 to 0.84).
- The probability of the first token being a tool call predicts whether a tool is needed (AUROC 0.96 to 0.98), better
  than asking the model.
- Gating keeps 66 to 70% of the thinking gain at 1.7B and 4B, and beats always thinking at 8B, at about half the
  tokens. Always thinking turns some correct declines into wrong calls; the gate doesn't think on declines.
- It changes only the requests it sends to thinking. The rest are exactly the direct answers.

The paper also reports twelve single-factor interventions on the same data: parameter descriptions matter a lot,
names barely; distractor tools hurt small models most; keyword retrieval can be worse than none while embedding
retrieval helps; forcing calls breaks abstention; schema repair rarely triggers.

## Reproduce

```bash
bash download_data.sh
pip install -e .
docker run --gpus all --ipc=host -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-8B --served-model-name local --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
python -m toolcall.run --label qwen3-8b       # all conditions, resumable journal in results/
python -m toolcall.signals --label qwen3-8b   # direct call with logprobs + probe
python -m toolcall.gate                       # signal AUROCs, budget curve, cross-validated gate
python -m toolcall.analyze                    # single-factor intervention tables
```

## Limitations

One model family, greedy decoding, single-turn BFCL only. Thinking adds 2 to 3 points overall on this data, which
bounds what any gate can gain; multi-turn agents may benefit more.

## Citation

```bibtex
@misc{ashter2026secondthought,
  title  = {SecondThought: Confidence-Gated Reasoning for Tool Calls in Small Local Models},
  author = {Ashter, Ali},
  year   = {2026},
  url    = {https://github.com/alleyxd/SecondThought}
}
```

## License

MIT for the code. BFCL data is Apache-2.0 and is downloaded, not included.
