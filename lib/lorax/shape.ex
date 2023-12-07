defmodule Lorax.Shape do
  @dummy_in_features 1

  @doc """
  ## Dense LoRA kernels
  Suppose we have a target node `W` and input tensor `x`,
  During injection, the A matrix will project the input tensor down to
  r-dimensional space. Afterwards, the B matrix will project the result back up
  to some unknown dimensionality. To figure out the output dimensionality,
  we inspect W's kernel shape.

  ## Convolution LoRA kernels
  In addition to figuring out the output dimensionality, we need to retrieve
  the convolution kernel_size of `W`. When calling Axon.conv, it's passed as an
  option, but then is no longer stored inside the Axon node. To figure out the
  kernel size, we also inspect W's kernel shape.
  """
  def calc_ab(op, r, parameters)

  def calc_ab(:dense, r, parameters) do
    shape_fn = get_kernel_shape_fn(parameters)
    shape = shape_fn.({nil, @dummy_in_features})
    out_features = elem(shape, Nx.rank(shape) - 1)

    a_shape_fn = fn x_shape ->
      {r, elem(x_shape, Nx.rank(x_shape) - 1)}
    end

    b_shape_fn = fn _x_shape ->
      {out_features, r}
    end

    {a_shape_fn, b_shape_fn}
  end

  def calc_ab(:conv, r, parameters) do
    shape_fn = get_kernel_shape_fn(parameters)

    {kernel_size, _kernel_size, _input_channels, output_filters} =
      shape_fn.({nil, nil, nil, @dummy_in_features})

    a_shape_fn = fn x_shape ->
      rank = Nx.rank(x_shape)
      in_features = x_shape |> elem(rank - 1)
      {kernel_size, kernel_size, in_features, r}
    end

    b_shape_fn = fn _x_shape ->
      {1, 1, r, output_filters}
    end

    {a_shape_fn, b_shape_fn}
  end

  defp get_kernel_shape_fn(parameters) do
    %Axon.Parameter{shape: shape_fn} =
      Enum.find(parameters, fn %Axon.Parameter{name: name} ->
        name == "kernel"
      end)

    shape_fn
  end
end
