defmodule Lorax do
  import Nx.Defn

  defmodule Config do
    defstruct r: 1,
              alpha: 2,
              dropout: 0.0,
              target_query: true,
              target_key: false,
              target_value: true,
              target_node_fn: nil
  end

  @moduledoc """
  Documentation for `Lorax`.
  """

  # 1. Move target node into a dummy container
  # 2. Make dummy be the input into lora
  # 3. Have lora node replaces target's position
  #
  # Beginning
  # ======
  # parent (id: 0) -> target (id: 1)
  #
  # Create dummy container
  # ======
  # parent (id: 0) -> target (id: 1) -> dummy (id: 2)
  #
  # Create lora node with input ids = [0, 2]
  # ======
  # parent (id: 0) -> target (id: 1) -> dummy (id: 2) -> lora (id: 3)
  #                ------------------------------------> lora (id: 3)
  #
  # target takes dummy's id, throw away dummy
  # =====
  # parent (id: 0) ->          target (id: 2)         -> lora (id: 3)
  #                ------------------------------------> lora (id: 3)
  #
  # lora takes target's original id
  # =====
  # parent (id: 0) ->          target (id: 2)          -> lora (id: 1)
  #                -------------------------------------> lora (id: 1)
  #
  # lora and target have now been swapped
  # any downstream node whose input was 1, will now take the lora values
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
      lora_node = create_lora_node(parent_axons, dummy_axon, config)
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

  defp create_lora_node(parent_axons, dummy_axon, %Config{
         r: r,
         alpha: alpha,
         dropout: dropout
       }) do
    scaling = alpha / r
    lora_A = Axon.param("lora_a", &dense_kernel_a(&1, &2, r), initializer: :normal)
    lora_B = Axon.param("lora_b", &dense_kernel_b(&1, &2, r), initializer: :zeros)

    # Send x, dummy Wx, and new params to create a new lora layer node
    Axon.layer(&lora_impl/5, parent_axons ++ [dummy_axon, lora_A, lora_B],
      op_name: :lora,
      dropout: dropout,
      scaling: scaling
    )
    |> then(fn %Axon{output: lora_id, nodes: lora_nodes} ->
      # Extract out the node, throwaway the Axon container
      %Axon.Node{} = lora_nodes[lora_id]
    end)
  end

  defn lora_impl(x, wx, lora_A, lora_B, opts \\ []) do
    dropout = opts[:dropout]
    scaling = opts[:scaling]

    x = Axon.Layers.dropout(x, Nx.Random.key(1337), rate: dropout)
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
      %Axon.Node{id: id, name: name_fn, op: :dense}, acc ->
        shortname =
          name_fn.(:dense, nil)
          |> String.split(".")
          |> List.last()

        if (target_key and shortname == "key") or
             (target_query and shortname == "query") or
             (target_value and shortname == "value") do
          [id | acc]
        else
          acc
        end

      %Axon.Node{}, acc ->
        acc
    end)
  end
end
