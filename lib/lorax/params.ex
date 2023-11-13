defmodule Lorax.Params do
  @moduledoc """
  Helper module for loading, downloading, filtering, and calculating the size of Axon parameters
  """

  @doc """
  Returns LoRA only params from a merged param map
  """
  def filter(lora_merged_params, original_params) do
    original_keys = Map.keys(original_params)
    Map.drop(lora_merged_params, original_keys)
  end

  @doc """
  Loads parameters from file path
  """
  def file_load!(params_path) do
    File.read!(params_path)
    |> Nx.deserialize()
  end

  @doc """
  Creates Kino cell for uploading serialized params file
  Must be placed in the last line of a Livebook cell.
  """
  def kino_file_load!(%Kino.Input{} = kino_input) do
    value = Kino.Input.read(kino_input)

    case value do
      nil ->
        raise "No param file uploaded"

      value ->
        path = Kino.Input.file_path(value.file_ref)

        try do
          file_load!(path)
        rescue
          ArgumentError -> raise "Invalid param file"
        end
    end
  end

  @doc """
  Creates Kino cell for downloading params map.
  Must be placed in the last line of a Livebook cell
  """
  def kino_download(
        params,
        filename \\ "params.lorax",
        label \\ "Download Params"
      ) do
    iodata = Nx.serialize(params)
    binary = IO.iodata_to_binary(iodata)

    Kino.Download.new(
      fn -> binary end,
      filename: filename,
      label: label
    )
  end

  @doc """
  Calculates total bytes of the tensors inside a parameter map
  """
  def size(%{} = params) do
    Enum.reduce(params, 0, fn {_k, v}, param_size ->
      layer_param_size =
        Enum.reduce(v, 0, fn {_layer_name, tensor}, acc -> acc + Nx.size(tensor) end)

      param_size + layer_param_size
    end)
  end
end
