defmodule Lorax.Conversion do
  @doc """
  Reads a safetensor file and returns a best-guess config for injection, and lora params
  """
  def read(path) do
    params = Safetensors.read!(path)

    #     - query, key, value
    # - cross_attention.output
    # - input_projection
    # - output_projection
    # - ffn.intermediate
    # - ffn.output
    # - conv_1
    # - conv_2
    # - shortcut.projection
    # - downsamples.{m}.conv (just conv is fine)
    # - upsamples.{m}.conv (just conv is fine)
    # - timestep_projection

    config = %Lorax.Config{
      r: calc_r(params),
      alpha: calc_alpha(params)
    }

    config
  end

  defp calc_r(params) do
    lora_down_keys =
      params
      |> Map.keys()
      |> Enum.filter(fn x -> String.contains?(x, "lora_down") end)

    first_dimensions =
      Enum.map(lora_down_keys, fn k ->
        params[k]
        |> Nx.shape()
        |> elem(0)
      end)

    lora_ranks = Enum.uniq(first_dimensions)

    are_ranks_same = length(lora_ranks) == 1

    if are_ranks_same do
      List.first(lora_ranks)
    else
      {:error, "Lorax does not support multiple ranks at the moment"}
    end
  end

  defp calc_alpha(params) do
    unique_alphas =
      params
      |> Map.keys()
      |> Enum.filter(fn key -> String.contains?(key, "alpha") end)
      |> Enum.map(fn key -> params[key] end)
      |> Enum.uniq()

    all_same = unique_alphas |> length() == 1

    if all_same do
      List.first(unique_alphas)
    else
      {:error, "Lorax does not support multiple alphas"}
    end
  end
end
