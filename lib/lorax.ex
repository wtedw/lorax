defmodule Lorax do
  @moduledoc """
  Simple Low-Rank Adaptation (LoRA) implementation

  ## LoRA model creation
  To create a LoRA model, freeze an existing model and inject LoRA layers using `Lorax.inject/2`.

  ```
  lora_model =
    model
    |> Axon.freeze()
    |> Lorax.inject(%Lorax.Config{
      r: 2,
      alpha: 4,
      dropout: 0.05,
      target_key: false,
      target_query: false,
      target_value: true
    })
  ```

  For more detailed guides, see
  1. [Finetuning LLMs with LoRA](finetuning_gpt_with_lora.livemd)
  2. [Running LLMs with LoRA](running_gpt_with_lora.livemd)


  LoRA layers are implemented by injecting new nodes into the Axon struct.
  These LoRA nodes represent the B and A matrices. Each node takes an input `x` and computes `BAx`.
  Furthermore, the LoRA node will receive `Wx` as an input and compute `Wx + BAx`.
  This isn't the standard implementation, but it simplifies the injection process.

  ## Injection Process

  Beginning state
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> B[target id:1]
  </div>

  Create an empty dummy node
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> B[target id:1] --> C[dummy id:2]
  </div>

  Create lora node with input ids = [0, 2]
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> B[target id:1] --> C[dummy id:2] --> E[lora id:3]
    A[input id:0] --> E[lora id:3]
  </div>

  target takes dummy's id, throw away dummy node
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> C[target id:2]
    C[target id:2] --> E[lora id:3]
    A[input id:0] --> E[lora id:3]
  </div>


  lora takes target's original id
  <div class="mermaid">
  flowchart LR
    A[input id:0] --> C[target id:2] --> E[lora id:1]
    A[input id:0] --> E[lora id:1]
  </div>


  lora and target are now swapped.
  Any downstream node that relied on node id:1 will now receive `Wx + BAx`
  """

  import Nx.Defn

  defmodule Config do
    @moduledoc """
    Config for `Lorax.inject/2`

    `r` is the rank in the low-rank matrices used in LoRA.
    A higher value of r increases the expressiveness of the adaptation,
    However, it also increases the number of parameters and the computational
    cost. Conversely, a lower value of r makes the adaptation simpler and less
    resource-intensive. Defaults to 1.

    `alpha` is a scaling factor that controls the magnitude of changes introduced
    by the low-rank matrices. A higher value of `alpha` means that the
    modifications made by LoRA have a greater impact on the model's original
    weights. This can lead to more significant changes in the model's behavior.
    A lower value results in more subtle changes. Defaults to 2.

    `dropout` specifies the dropout rate applied to the low-rank matrices.

    `dropout_seed` determines the seed used for `Nx.Random.key/1` during
    dropout application. When defined, it ensures that the LoRA adapter
    produces consistent tensor values, assuming that other layers also have
    deterministic outputs.

    `param_type` specifies the numerical representation for the A and B
    matrices. Defaults to float32

    `target_query` specifies whether to apply LoRA to all query matrices in an
    attention block. Defaults to true.

    `target_value` specifies whether to apply LoRA to all value matrices in an
    attention block. Defaults to true.

    `target_key` specifies whether to apply LoRA to all key matrices in an
    attention block. Defaults to true.
    """
    defstruct r: 1,
              alpha: 2,
              dropout: 0.0,
              dropout_seed: nil,
              param_type: {:f, 32},
              node_name_fn: nil,
              target_query: true,
              target_key: true,
              target_value: true,
              target_node_fn: nil
  end

  @doc """
  Returns a modified Axon model with LoRA nodes inserted according to the provided configuration.

  `target_key`, `target_query`, `target_value` are required if `target_node_fn` isn't specified

  ## Examples
  ```
  lora_model =
    model
    |> Axon.freeze()
    |> Lorax.inject(%Lorax.Config{
      r: 2,
      alpha: 4,
      dropout: 0.05,
      target_key: true,
      target_query: true,
      target_value: true
    })
  ```

  ## Targeting nodes manually
  ```
  lora_model =
    model
    |> Axon.freeze()
    |> Lorax.inject(%Lorax.Config{
      r: 2,
      alpha: 4,
      dropout: 0.05,
      target_node_fn: fn %Axon.Node{name: name_fn} ->
        # names are generated lazily, and look like "decoder.blocks.11.self_attention.value"
        # have to invoke the function to see what layer the node represents
        # https://github.com/elixir-nx/axon/blob/v0.6.0/lib/axon.ex#L3923
        name = name_fn.(nil, nil)
        shortname = String.split(name, ".") |> List.last()

        if shortname == "output" do
          true
        else
          false
        end
      end
    })
  ```
  """
  def inject(%Axon{} = axon, %Config{} = config) do
    target_nodes = get_target_nodes(axon, config)

    Enum.reduce(target_nodes, axon, fn target_id, %Axon{nodes: acc_nodes} = acc ->
      # Grab our target node, create a fake Axon container for it
      target_node = acc_nodes[target_id]
      target_axon = %Axon{acc | output: target_id}

      # Get its parent and create fake Axon containers
      # Note: The parent field of Axon.Node is usually a list,
      #       but for our purposes, it's just a list of one input
      parent_ids = target_node.parent
      parent_axons = Enum.map(parent_ids, fn id -> %Axon{acc | output: id} end)

      # Create a dummy Axon container for target to move into
      dummy_axon = %Axon{output: dummy_id} = Axon.nx(target_axon, fn x -> x end)

      # lora node takes target's place
      # target node takes dummy's place
      lora_node = create_lora_node(parent_axons, dummy_axon, target_node.name, config)
      lora_node = %Axon.Node{lora_node | id: target_id}
      target_node = %Axon.Node{target_node | id: dummy_id}

      # update Axon container's map of nodes so that
      # 1. whenever downstream nodes reference target_id, it'll now point to our lora node
      # 2. whenever lora node references dummy id, it'll take the output value (Wx) from target
      new_nodes =
        acc_nodes
        |> Map.put(target_id, lora_node)
        |> Map.put(dummy_id, target_node)

      %Axon{acc | nodes: new_nodes}
    end)
  end

  # Parent + dummy axon are inputs to create the lora node
  # target_node_name_fn is provided to help create a name for our new lora node
  defp create_lora_node(parent_axons, dummy_axon, target_name_fn, %Config{
         r: r,
         alpha: alpha,
         dropout: dropout,
         dropout_seed: dropout_seed,
         param_type: param_type
       }) do
    scaling = alpha / r
    dropout_seed = dropout_seed || :erlang.system_time()

    lora_A =
      Axon.param("lora_a", &dense_kernel_a(&1, &2, r),
        initializer: :normal,
        type: param_type
      )

    lora_B =
      Axon.param("lora_b", &dense_kernel_b(&1, &2, r),
        initializer: :zeros,
        type: param_type
      )

    name_fn = fn (op, op_count) ->
      target_name = target_name_fn.(op, op_count)
      IO.inspect(op, label: "what is op?")
      IO.inspect(op_count, label: "what is op count?")

      IO.inspect(target_name, label: "parent name")
      lora_name = "lora_" <> target_name
      |> IO.inspect(label: "lora name")
    end

    Axon.layer(&lora_impl/5, parent_axons ++ [dummy_axon, lora_A, lora_B],
      op_name: :lora,
      name: name_fn,
      dropout: dropout,
      dropout_seed: dropout_seed,
      scaling: scaling
    )
    |> then(fn %Axon{output: lora_id, nodes: lora_nodes} ->
      # Extract out the node, throwaway the Axon container
      %Axon.Node{} = lora_nodes[lora_id]
    end)
  end

  defnp lora_impl(x, wx, lora_A, lora_B, opts \\ []) do
    dropout = opts[:dropout]
    dropout_seed = opts[:dropout_seed]
    scaling = opts[:scaling]

    x = Axon.Layers.dropout(x, Nx.Random.key(dropout_seed), rate: dropout)
    after_a = Axon.Layers.dense(x, lora_A |> Nx.transpose())
    after_b = Nx.dot(after_a, lora_B |> Nx.transpose())
    bax = Nx.multiply(after_b, scaling)

    Nx.add(wx, bax)
  end

  defp dense_kernel_a(x_shape, _wx_shape, r) do
    {r, elem(x_shape, Nx.rank(x_shape) - 1)}
  end

  defp dense_kernel_b(_x_shape, wx_shape, r) do
    {elem(wx_shape, Nx.rank(wx_shape) - 1), r}
  end

  defp get_target_nodes(axon, %Config{target_node_fn: target_node_fn})
       when is_function(target_node_fn, 1) do
    Axon.reduce_nodes(axon, [], fn %Axon.Node{id: id} = node, acc ->
      if target_node_fn.(node) do
        [id | acc]
      else
        acc
      end
    end)
  end

  defp get_target_nodes(
         axon,
         %Config{
           target_query: target_query,
           target_key: target_key,
           target_value: target_value
         }
       ) do
    Axon.reduce_nodes(axon, [], fn
      %Axon.Node{id: id, name: name_fn, op: :dense, op_name: op_name}, acc ->
        shortname =
          name_fn.(:dense, nil)
          |> String.split(".")
          |> List.last()

        IO.inspect(shortname, label: "shortname")
        IO.inspect(op_name, label: "op name")

        if (target_key and shortname == "key") or
             (target_query and shortname == "query") or
             (target_value and shortname == "value") do
          IO.inspect("here")
          [id | acc]
        else
          acc
        end

      %Axon.Node{}, acc ->
        acc
    end)
  end
end
