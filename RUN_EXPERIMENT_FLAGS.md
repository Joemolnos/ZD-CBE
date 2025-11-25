# `run_experiment.py` Command-Line Flags

- **`--config <path>`**
  Loads experiment settings from the JSON file at `<path>`. Default: `config.json`.

- **`--dream_steps <int>`**
  Sets the number of steps for the Dreaming phase. Default: `5000`.

- **`--awakening_epochs <int>`**
  Sets the number of epochs for the Awakening phase. Default: `100`.

- **`--batch_size <int>`**
  Specifies the batch size used during Awakening training. Default: `32`.

- **`--learning_rate <float>`**
  Overrides the learning rate used when initializing the Dream phase optimizer. Default: `0.001`.

- **`--no_visualization`**
  Disables real-time visualizations by forcing `visualization_interval` to zero.

- **`--dream_checkpoint <path>`**
  Loads the Dreamer model weights from `<path>` prior to launching the experiment.

- **`--dream_metrics <path>`**
  Supplies a previously saved Dream phase metrics pickle to recreate Dream analytics without rerunning the phase.

- **`--skip_dream_phase`**
  Skips the Dream phase entirely and uses the checkpoint and metrics provided via the flags above.
