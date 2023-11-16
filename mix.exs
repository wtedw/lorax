defmodule Lorax.MixProject do
  use Mix.Project

  def project do
    [
      app: :lorax,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: [
        extras: [
          "README.md",
          "guides/finetuning_gpt_with_lora.livemd",
          "guides/running_gpt_with_lora.livemd",
        ],
        main: "readme",
        before_closing_body_tag: fn
          :html ->
            """
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({startOnLoad: true})</script>
            """
          _ -> ""
        end
      ]
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
      {:axon, "~> 0.6.0"},
      {:nx, "~> 0.6.0"},
      {:kino, "~> 0.11.0"},
    ]
  end
end
