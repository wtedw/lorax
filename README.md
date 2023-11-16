# Lorax

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

In practice, every model has some unique architecture that you need to account for.
For more detailed guides, see

1. [Finetuning LLMs with LoRA](finetuning_gpt_with_lora.livemd)
2. [Running LLMs with LoRA](running_gpt_with_lora.livemd)

## Default Settings

The default config applies LoRA to all query and value matrices. r = 1, alpha = 2.

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

While LoRA significantly reduces the GPU requirements for fine-tuning, using LoRA on LLMs that are bigger than GPT2 still requires a GPU with high vRAM.

Most of the examples here were fine-tuned on an Nvidia T4 / A10G on Huggingface Spaces. Attempting to fine-tune Mistral 7B on Huggingface's A10x4 (the largest available w/ 96 vRAM) will cause cuda OOM errors. Further work needs to be done to reduce the memory usage on GPUs (like implementing QLoRA).