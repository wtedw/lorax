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
      lora_node = create_lora_node(target_node, parent_axons, dummy_axon, config)
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

  defp get_kernel_shape(op, r, parameters) do
    case op do
      :conv ->
        kernel_size = get_kernel_size(parameters)
        {&conv_kernel_a(&1, &2, r, kernel_size), &conv_kernel_b(&1, &2, r)}

      :dense ->
        {&dense_kernel_a(&1, &2, r), &dense_kernel_b(&1, &2, r)}
        |> IO.inspect(label: "shape")
    end
  end

  def inject2(%Axon{} = model, %Config{r: r} = config) do
    %MapSet{} = target_nodes = get_target_nodes2(model, config)

    Axon.map_nodes(model, fn
      %Axon.Node{id: id, op: op, parameters: parameters} = axon_node
      when is_map_key(target_nodes.map, id) ->
        shape = get_kernel_shape(op, r, parameters)
        lora_a = Axon.param("lora_a", shape, initializer: :normal)
        lora_b = Axon.param("lora_b", shape, initializer: :zeros)

        lora_dropout_key =
          Axon.param("lora_dropout_key", shape, initializer: :zeros, type: :u32)

        Axon.wrap_node(axon_node, [lora_a, lora_b, lora_dropout_key], &lora_impl2/4,
          injected_key: "lora_params"
        )

      axon_node ->
        axon_node
    end)
  end

  defp get_kernel_size(params) do
    %Axon.Parameter{shape: shape} =
      Enum.find(params, fn %Axon.Parameter{name: name} ->
        name == "kernel"
      end)

    kernel_shape = shape.({nil, nil, nil, 1})
    elem(kernel_shape, 0)
  end

  deftransform lora_impl2(
                 [input, w_kernel],
                 forward,
                 %{
                   "lora_a" => lora_a,
                   "lora_b" => lora_b,
                   "lora_dropout_key" => key
                 },
                 opts \\ []
               ) do
    dropout = opts[:dropout]
    scaling = opts[:scaling]
    mode = opts[:mode]

    x = input
    wx = forward.(input, w_kernel, mode: mode)

    {x, next_key} =
      case mode do
        :inference -> {x, :ignored}
        :train -> Axon.Layers.dropout(x, key, rate: dropout)
      end

    after_a = Axon.Layers.dense(x, lora_a |> Nx.transpose())
    after_b = Nx.dot(after_a, lora_b |> Nx.transpose())
    bax = Nx.multiply(after_b, scaling)
    out = Nx.add(wx, bax)

    case mode do
      :inference ->
        out

      :train ->
        %Axon.StatefulOutput{
          output: out,
          state: %{"lora_dropout_key" => next_key}
        }
    end
  end

  defp create_lora_node(
         %Axon.Node{name: target_name_fn, op: :conv, opts: opts},
         parent_axons,
         dummy_axon,
         %Config{
           r: r,
           alpha: alpha,
           dropout: dropout,
           dropout_seed: dropout_seed,
           param_type: param_type
         }
       ) do
    scaling = alpha / r
    dropout_seed = dropout_seed || :erlang.system_time()

    # todo
    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]

    IO.inspect(opts, label: "creating lora node w/ opts")

    lora_A =
      Axon.param("lora_down", &conv_kernel_a(&1, &2, r, kernel_size),
        initializer: :normal,
        type: param_type
      )

    # todo
    lora_B =
      Axon.param("lora_up", &conv_kernel_b(&1, &2, r),
        initializer: :zeros,
        type: param_type
      )

    lora_name_fn = create_name_fn(target_name_fn)

    Axon.layer(&lora_conv_impl/5, parent_axons ++ [dummy_axon, lora_A, lora_B],
      op_name: :lora,
      name: lora_name_fn,
      dropout: dropout,
      dropout_seed: dropout_seed,
      scaling: scaling,
      kernel_size: kernel_size,
      strides: strides,
      padding: padding
    )
    |> then(fn %Axon{output: lora_id, nodes: lora_nodes} ->
      # Extract out the node, throwaway the Axon container
      %Axon.Node{} = lora_nodes[lora_id]
    end)
  end

  # Parent + dummy axon are inputs to create the lora node
  # target_node_name_fn is provided to help create a name for our new lora node
  defp create_lora_node(
         %Axon.Node{name: target_name_fn, op: _target_op},
         parent_axons,
         dummy_axon,
         %Config{
           r: r,
           alpha: alpha,
           dropout: dropout,
           dropout_seed: dropout_seed,
           param_type: param_type
         }
       ) do
    scaling = alpha / r
    dropout_seed = dropout_seed || :erlang.system_time()

    lora_A =
      Axon.param("lora_down", &dense_kernel_a(&1, &2, r),
        initializer: :normal,
        type: param_type
      )

    lora_B =
      Axon.param("lora_up", &dense_kernel_b(&1, &2, r),
        initializer: :zeros,
        type: param_type
      )

    lora_name_fn = create_name_fn(target_name_fn)

    Axon.layer(&lora_impl/5, parent_axons ++ [dummy_axon, lora_A, lora_B],
      op_name: :lora,
      name: lora_name_fn,
      dropout: dropout,
      dropout_seed: dropout_seed,
      scaling: scaling
    )
    |> then(fn %Axon{output: lora_id, nodes: lora_nodes} ->
      # Extract out the node, throwaway the Axon container
      %Axon.Node{} = lora_nodes[lora_id]
    end)
  end

  defnp lora_conv_impl(x, wx, lora_A, lora_B, opts \\ []) do
    scaling = opts[:scaling]

    # kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    conv_opts = [strides: strides, padding: padding]

    after_a = Axon.Layers.conv(x, lora_A, conv_opts)
    # |> print_expr(label: "after a")
    after_b = Axon.Layers.conv(after_a, lora_B)
    # |> print_expr(label: "after b")
    bax = Nx.multiply(after_b, scaling)
    # |> print_expr(label: "bax")
    Nx.add(wx, bax)

    # Apparently we can just fuse the kernels, so can just add the lora_A, lora_B
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

  defp create_name_fn(target_name_fn) do
    fn op, op_count ->
      target_name = target_name_fn.(op, op_count)

      ("lora_" <> target_name)
      |> IO.inspect(label: "lora node name")
    end
  end

  # lora down, projects down to r channels
  # the channels is equal to the # of in_features
  # which we can determine with x_shape
  defp conv_kernel_a(x_shape, wx_shape, r, kernel_size) do
    IO.inspect(x_shape, label: "conv_kernel_a x")
    IO.inspect(wx_shape, label: "conv_kernel_a wx")

    rank = Nx.rank(x_shape)
    in_features = x_shape |> elem(rank - 1)
    # target node has shape that looks like this f32[3][3][320][320]
    {kernel_size, kernel_size, in_features, r}
    |> IO.inspect(label: "convkernela")
  end

  # lora up, the channels is equal to the out_features
  # which we can determine by wx
  defp conv_kernel_b(x_shape, wx_shape, r) do
    IO.inspect(x_shape, label: "conv_kernel_b x")
    IO.inspect(wx_shape, label: "conv_kernel_b wx")

    rank = Nx.rank(wx_shape)
    out_features = wx_shape |> elem(rank - 1)
    # target node has shape that looks like this f32[3][3][320][320]
    {1, 1, r, out_features}
    |> IO.inspect(label: "convkernelb")
  end

  defp dense_kernel_a(x_shape, wx_shape, r) do
    IO.inspect(x_shape, label: "dense_kernel_a x")
    IO.inspect(wx_shape, label: "dense_kernel_a wx")

    {r, elem(x_shape, Nx.rank(x_shape) - 1)}
  end

  defp dense_kernel_b(x_shape, wx_shape, r) do
    IO.inspect(x_shape, label: "dense_kernel_b x")
    IO.inspect(wx_shape, label: "dense_kernel_b wx")

    {elem(wx_shape, Nx.rank(wx_shape) - 1), r}
  end

  defp get_target_nodes2(axon, %Config{target_node_fn: target_node_fn})
       when is_function(target_node_fn, 1) do
    Axon.reduce_nodes(axon, MapSet.new(), fn %Axon.Node{id: id} = node, map_set ->
      if target_node_fn.(node) do
        MapSet.put(map_set, id)
      else
        map_set
      end
    end)
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

  defp calc_shortname(%Axon.Node{name: name_fn}) do
    name_fn.(nil, nil)
    |> String.split(".")
    |> List.last()
  end
end
