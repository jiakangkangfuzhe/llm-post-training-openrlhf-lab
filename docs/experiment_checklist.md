# Experiment checklist

Before running:

- [ ] record model name, dataset path, commit hash, GPU type/count;
- [ ] save full training script and config;
- [ ] confirm tokenizer/chat template;
- [ ] confirm max prompt/completion length;
- [ ] run answer extractor self-test;
- [ ] set output/log directories.

During running:

- [ ] monitor GPU memory;
- [ ] track step time and tokens/sec if available;
- [ ] watch reward / KL / loss trends;
- [ ] save failed samples and truncation cases.

After running:

- [ ] parse logs with `src/log_parser.py`;
- [ ] write `results/<run_name>.md`;
- [ ] commit config + script + summary, not huge checkpoints.
