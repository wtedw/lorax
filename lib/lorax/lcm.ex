defmodule Lorax.Lcm do
  @doc """
  Loads some serialized LCM LoRA data and returns a returns a tuple of `{%Lorax.Config{}, lora_params}`
  """
  def load!(data) do
    params = Safetensors.load!(data)
    create_config_params(params)
  end

  @doc """
  Reads some LCM LoRA safetensor file and returns a tuple of `{%Lorax.Config{}, lora_params}`
  """
  def read!(path) do
    params = Safetensors.read!(path)
    create_config_params(params)
  end

  defp create_config_params(params) do
    config = %Lorax.Config{
      r: calc_r(params),
      alpha: calc_alpha(params),
      param_type: calc_param_type(params),
      target_node_fn: calc_target_node_fn(params)
    }

    params =
      Enum.reduce(params, %{}, fn {layer_name, tensor}, acc ->
        layer_name = translate_kohya_layer(layer_name)
        split = String.split(layer_name, ".")

        # last is either alpha or weight
        # if last is weight, then last2 is lora_down or lora_up
        [last, last2 | _] = Enum.reverse(split)

        if last == "alpha" do
          # Skip this iteration as we don't want anything related to 'alpha'
          acc
        else
          new_layer_name = Enum.drop(split, -2) |> Enum.join(".")
          param_name = last2

          # it's a convolution kernel
          tensor =
            if Nx.rank(tensor) == 4 do
              Nx.transpose(tensor)
            else
              tensor
            end

          case Map.get(acc, new_layer_name) do
            nil ->
              Map.put(acc, new_layer_name, %{param_name => tensor})

            current_params ->
              new_params = Map.put(current_params, param_name, tensor)
              Map.put(acc, new_layer_name, new_params)
          end
        end
      end)

    {config, params}
  end

  defp calc_r(params) do
    lora_ranks =
      params
      |> Map.keys()
      |> Enum.filter(fn x -> String.contains?(x, "lora_down") end)
      |> Enum.map(fn k -> params[k] |> Nx.shape() |> elem(0) end)
      |> Enum.uniq()

    if length(lora_ranks) == 1 do
      List.first(lora_ranks)
    else
      {:error, "Invalid LCM LoRA params. r values should all be the same"}
    end
  end

  defp calc_alpha(params) do
    lora_alphas =
      params
      |> Map.keys()
      |> Enum.filter(fn key -> String.contains?(key, "alpha") end)
      |> Enum.map(fn key -> params[key] end)
      |> Enum.map(fn tensor -> Nx.to_number(tensor) end)
      |> Enum.uniq()

    if length(lora_alphas) == 1 do
      lora_alphas |> List.first() |> Nx.to_number()
    else
      {:error, "Invalid LCM LoRA params. alpha values should all be the same"}
    end
  end

  defp calc_param_type(params) do
    lora_keys =
      params
      |> Map.keys()
      |> Enum.filter(fn key -> String.contains?(key, "lora_down") end)

    random_tensor = params[List.first(lora_keys)]
    Nx.type(random_tensor)
  end

  defp calc_target_node_fn(_params) do
    fn %Axon.Node{name: name_fn} ->
      split =
        name_fn.(nil, nil)
        |> String.split(".")

      shortname = split |> List.last()

      twoname =
        if length(split) >= 2 do
          [last, last2 | _rest] = Enum.reverse(split)
          last2 <> "." <> last
        else
          nil
        end

      target_shortnames = [
        "query",
        "key",
        "value",
        "input_projection",
        "output_projection",
        "conv_1",
        "conv_2",
        "projection",
        "conv",
        "timestep_projection"
      ]

      target_twonames = [
        "self_attention.output",
        "cross_attention.output",
        "ffn.intermediate",
        "ffn.output"
      ]

      shortname in target_shortnames or twoname in target_twonames
    end
  end

  defp translate_kohya_layer(layer_name) do
    layer_name
    |> String.replace("lora_unet_down_blocks_", "lora_down_blocks.")
    |> String.replace("lora_unet_up_blocks_", "lora_up_blocks.")
    |> String.replace("lora_unet_mid_block_attentions_", "lora_mid_block.transformers.")
    |> String.replace("lora_unet_mid_block_resnets_", "lora_mid_block.residual_blocks.")
    |> String.replace("_attentions_", ".transformers.")
    |> String.replace("_transformer_blocks_", ".blocks.")
    |> String.replace("_downsamplers_", ".downsamples.")
    |> String.replace("_upsamplers_", ".upsamples.")
    |> String.replace("_resnets_", ".residual_blocks.")
    |> String.replace("_attn1_to_q", ".self_attention.query")
    |> String.replace("_attn1_to_k", ".self_attention.key")
    |> String.replace("_attn1_to_v", ".self_attention.value")
    |> String.replace("_attn1_to_out_0", ".self_attention.output")
    |> String.replace("_attn2_to_q", ".cross_attention.query")
    |> String.replace("_attn2_to_k", ".cross_attention.key")
    |> String.replace("_attn2_to_v", ".cross_attention.value")
    |> String.replace("_attn2_to_out_0", ".cross_attention.output")
    |> String.replace("_proj_in", ".input_projection")
    |> String.replace("_proj_out", ".output_projection")
    |> String.replace("_ff_net_0_proj", ".ffn.intermediate")
    |> String.replace("_ff_net_2", ".ffn.output")
    |> String.replace("_conv1", ".conv_1")
    |> String.replace("_conv2", ".conv_2")
    |> String.replace("_conv_shortcut", ".shortcut.projection")
    # Make sure to replace conv1, conv2, and conv_shortcut before generic conv
    |> String.replace("_conv", ".conv")
    |> String.replace("_time_emb_proj", ".timestep_projection")
  end
end
