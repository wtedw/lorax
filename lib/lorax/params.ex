defmodule Lorax.Params do
  def param_size(%{} = params) do
    Enum.reduce(params, 0, fn {_k, v}, param_size ->
      layer_param_size =
        Enum.reduce(v, 0, fn {_layer_name, tensor}, acc -> acc + Nx.size(tensor) end)

      param_size + layer_param_size
    end)
  end

  def serialize(lora_params, original_params, serialize_opts \\ []) do
    original_keys = Map.keys(original_params)
    lora_params = Map.drop(lora_params, original_keys)
    Nx.serialize(lora_params, serialize_opts)
  end

  def file_load!(params_path) do
    File.read!(params_path)
    |> Nx.deserialize()
  end

  def merge_params(lora_params, og_params) do
    Map.merge(og_params, lora_params)
  end

  def kino_download(
        lora_params,
        original_params,
        filename \\ "params.lorax",
        label \\ "Lora Params"
      ) do
    iodata = serialize(lora_params, original_params)
    binary = IO.iodata_to_binary(iodata)

    Kino.Download.new(
      fn -> binary end,
      filename: filename,
      label: label
    )
  end

  # This should probably be a Kino smart cell or something
  # Note: This only returns the LoRA params, to run a model, you need to merge the original params
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
end
