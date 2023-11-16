# Lorax

This package implements Low-Rank Adaptation (LoRA), a popular method for fine-tuning large language models.

![lora diagram](https://raw.githubusercontent.com/spawnfest/lorax/main/diagram.png)

## Installation

This package can be installed by adding `lorax` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:lorax, "~> 0.1.0"}
    # or
    {:lorax, git: "https://github.com/wtedw/lorax.git"},
  ]
end
```

## How To Fine-tune an LLM

In general,

1. Import your model
2. Inject trainable LoRA parameters
3. Train LoRA model
4. Download LoRA only params

```
{:ok, model_info} = Bumblebee.load_model({:hf, "gpt2"})
%{model: gpt2_model, params: gpt2_params} = model_info

lora_model =
  gpt2_model
  |> Axon.freeze()
  |> Lorax.inject(%Lorax.Config{
      r: 2,
      alpha: 4,
      dropout: 0.05,
      target_query: true,
      target_key: true,
      target_value: true
  })

lora_merged_params =
  Axon.build(lora_model, mode: :train)
  |> Axon.Loop.trainer(custom_loss_fn, Polaris.Optimizers.adam(learning_rate: 3.0e-4))
  |> Axon.Loop.run(train_batch_stream, gpt2_params, epochs: 3, iterations: 1000, compiler: EXLA)

lora_params = lora_merged_params
  |> Lorax.Params.filter(gpt2_params)
  |> Lorax.Params.kino_download()
```

For more detailed guides, see

1. [Training LoRAs](finetuning_gpt_with_lora.livemd)
1. [Running LoRAs](running_gpt_with_lora.livemd)

## Default Settings

The default configs applies LoRA to all query and value matrices.
r is set to 1, alpha to 2.

The original LoRA paper found that configuring only query and value matrices was effective enough for fine-tuning. But in practice, r values of 2, 4, or 8 have more expressive capabilities.

## Recommended Settings

These settings are for an A10 small w/ 24gb vRAM

```
Lora Config
- r value  = at least 2
- alpha value = r * 2
- batch size = 4
- sequence_length = 512

Training
- learning_rate of 3.0e-4

Text Generation
- multinomial sampling
- p = 0.06 or 0.08 for more variety (or if you experience repetitive results)
```

## Limitations

While the LoRA algorithm significantly reduces the GPU requirements for fine-tuning a model, using LoRA on LLMs that are bigger than GPT2 still requires a GPU with high vRAM.

Most of the examples here were fine-tuned on an A10G on Huggingface Spaces. Attempting to fine-tune Mistral 7B on Huggingface's A10x4 (the largest available w/ 96 vRAM) will cause cuda OOM errors. To fine-tune on consumer GPUs, [quantization work](https://github.com/elixir-nx/axon/issues/100) needs to be done to implement the QLoRA algorithm.
