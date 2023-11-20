# Lorax

[![Module Version](https://img.shields.io/hexpm/v/lorax.svg)](https://hex.pm/packages/lorax)
[![Hex Docs](https://img.shields.io/badge/hex-docs-lightgreen.svg)](https://hexdocs.pm/lorax/)

This package implements [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685), a popular method for fine-tuning large language models.

## Installation

This package can be installed by adding `lorax` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:lorax, "~> 0.1.0"}
  ]
end
```

## Fine-tuning an LLM with LoRA

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
      dropout: 0.05
  })

lora_merged_params =
  Axon.build(lora_model, mode: :train)
  |> Axon.Loop.trainer(custom_loss_fn, Polaris.Optimizers.adam(learning_rate: 3.0e-4))
  |> Axon.Loop.run(train_batch_stream, gpt2_params, epochs: 3, iterations: 1000, compiler: EXLA)

lora_params = lora_merged_params
  |> Lorax.Params.filter(gpt2_params)
  |> Lorax.Params.kino_download()
```

In practice, every model has some unique architecture that you need to account for.
For more detailed guides, see

1. [Finetuning LLMs with LoRA](finetuning_gpt_with_lora.livemd)
2. [Running LLMs with LoRA](running_gpt_with_lora.livemd)

## Default Settings

The default config applies LoRA to all query, key, value matrices. r = 1, alpha = 2.

The LoRA paper demonstrated that adapting only the query and value matrices with r = 1 achieved effective fine-tuning results. However, for larger language models, people often choose much higher values of r and sometimes target all linear layers.

## Recommended Settings

These settings works well for fine-tuning smaller LLMs (~1b param models)

```
Lora Config
- r value  = at least 2
- alpha value = r * 2

Training
- learning_rate of 3.0e-4 with an adam optimizer

Text Generation
- multinomial sampling
- p = 0.06 or 0.08 for more variety (or if you experience repetitive results)
```

For more details on configuring LoRA hyperparameters, see this [post](https://lightning.ai/pages/community/lora-insights/) by Sebastian Raschka.

## Limitations

1. GPU Memory Requirements: Although LoRA reduces the GPU requirements for fine-tuning, larger LLMs beyond GPT2 still demand GPUs with substantial vRAM. Inadequate memory management can lead to cuda OOM crashes.

2. Fine-Tuning Speed: The training speed of this library isn't on par with Huggingface's PEFT library. Further optimizations can be done to close the gap.

Note: For minor fine-tuning tasks without a GPU, the BinaryBackend is a viable option, often resulting in smoother training runs. Future updates will focus on minimizing GPU memory usage by reducing the amount of tensors stored during training, and potentially a QLoRA implementation one day.
