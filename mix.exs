defmodule Lorax.MixProject do
  use Mix.Project

  @scm_url "https://github.com/wtedw/lorax"

  def project do
    [
      app: :lorax,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      description: "Simple LoRA implementation for fine-tuning Axon models",
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      # {:axon, "~> 0.6.0"},
      {:axon, path: "/Users/ted/CS/elixir/ex/axon"},
      {:nx, "~> 0.6.0"},
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
      {:kino, "~> 0.11.0"},
      {:safetensors, "~> 0.1.2"}
    ]
  end

  defp docs do
    [
      extras: [
        "README.md",
        "guides/finetuning_gpt_with_lora.livemd",
        "guides/running_gpt_with_lora.livemd"
      ],
      main: "readme",
      before_closing_body_tag: fn
        :html ->
          """
          <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
          <script>mermaid.initialize({startOnLoad: true})</script>
          """

        _ ->
          ""
      end
    ]
  end

  defp package do
    [
      maintainers: ["Ted Wong"],
      licenses: ["MIT"],
      links: %{"GitHub" => @scm_url}
    ]
  end
end
