Application.put_env(:sample, PhoenixDemo.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 8080],
  server: true,
  live_view: [signing_salt: "bumblebee"],
  secret_key_base: String.duplicate("b", 64)
)

Mix.install([
  {:plug_cowboy, "~> 2.6"},
  {:jason, "~> 1.4"},
  {:phoenix, "~> 1.7.10"},
  {:phoenix_live_view, "~> 0.20.1"},
  # Bumblebee and friends
  {:bumblebee, "~> 0.4.0"},
  {:nx, "~> 0.6.0"},
  {:exla, "~> 0.6.0"},
  {:lorax, git: "https://github.com/wtedw/lorax.git"},
  {:req, "~> 0.4.0"}
])

Application.put_env(:nx, :default_backend, EXLA.Backend)

defmodule PhoenixDemo.Layouts do
  use Phoenix.Component

  def render("live.html", assigns) do
    ~H"""
    <script src="https://cdn.jsdelivr.net/npm/phoenix@1.7.10/priv/static/phoenix.min.js">
    </script>
    <script
      src="https://cdn.jsdelivr.net/npm/phoenix_live_view@0.20.1/priv/static/phoenix_live_view.min.js"
    >
    </script>
    <script>
      const liveSocket = new window.LiveView.LiveSocket("/live", window.Phoenix.Socket);
      liveSocket.connect();
    </script>
    <script src="https://cdn.tailwindcss.com">
    </script>
    <%= @inner_content %>
    """
  end
end

defmodule PhoenixDemo.ErrorView do
  def render(_, _), do: "error"
end

defmodule PhoenixDemo.SampleLive do
  use Phoenix.LiveView, layout: {PhoenixDemo.Layouts, :live}

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(
       text: "<title>Elixir 5 released</title>",
       current_id: nil,
       current_content: "",
       current_likes: "",
       post_content: %{},
       label: nil,
       task: nil
     )
     |> stream(:posts, [])}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="h-screen w-screen flex items-center justify-center antialiased">
      <div class="flex flex-col h-1/2 w-1/2">
        <h1 class="text-4xl font-extrabold text-center mb-2">Elixir Thread Simulator</h1>
        <form phx-submit="predict" class="m-0 flex space-x-2">
          <input
            class="block w-full p-2.5 bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500"
            type="text"
            name="text"
            value={@text}
          />
          <button
            class="px-5 py-2.5 text-center mr-2 inline-flex items-center text-white bg-blue-700 font-medium rounded-lg text-sm hover:bg-blue-800 focus:ring-4 focus:ring-blue-300"
            type="submit"
            disabled={@task != nil}
          >
            Predict
          </button>
        </form>
        <div id="posts" phx-update="stream" class="flex flex-col flex-grow">
          <%= for {id, post} <- @streams.posts do %>
            <.post content={post.content} author={post.author} id={id} likes={post.likes} />
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  def post(assigns) do
    ~H"""
    <div id={@id} class="p-4">
    <div class="bg-[#f5eef6] rounded-lg shadow-md p-5">
      <div class="flex items-center mb-4">
          <p class="text-lg font-semibold"><%= @author %></p>
      </div>
      <p class="text-gray-800 mb-4">
        <%= @content %>
      </p>
      <div class="flex justify-between items-center">
        <span class="text-gray-600 text-sm"><%= @likes %></span>
      </div>
    </div>
    </div>
    """
  end

  @impl true
  def handle_event("predict", %{"text" => text}, socket) do
    liveview_pid = self()

    task =
      Task.async(fn ->
        stream = Nx.Serving.batched_run(PhoenixDemo.Serving, text)

        Enum.each(stream, fn token ->
          send(liveview_pid, {:new_token, token})
        end)
      end)

    {:noreply,
     socket
     |> assign(text: text, task: task)
     |> assign(current_content: "", current_author: "", current_id: nil, current_likes: "")
     |> stream(:posts, [], reset: true)}
  end

  def handle_info({:new_token, "<likes>" <> num}, socket) do
    if socket.assigns.current_id == nil do
      {:noreply, socket}
    else
      likes_string = if num == "1", do: "1 like", else: num <> " likes"

      current_author = socket.assigns.current_author
      current_id = socket.assigns.current_id
      current_content = socket.assigns.current_content

      {:noreply,
       socket
       |> assign(:current_likes, likes_string)
       |> stream_insert(
         :posts,
         %{id: current_id, author: current_author, content: current_content, likes: likes_string},
         at: -1
       )}
    end
  end

  def handle_info({:new_token, " like</likes>\n"}, socket), do: {:noreply, socket}
  def handle_info({:new_token, " likes</likes>\n"}, socket), do: {:noreply, socket}

  # note: This isn't perfect, sometimes the author's name is split into two tokens
  def handle_info({:new_token, "<author>" <> rest}, socket) do
    name = String.split(rest, "</author>") |> List.first()
    unique_id = System.unique_integer([:positive])

    {:noreply,
     socket
     |> stream_insert(:posts, %{id: unique_id, author: name, content: "", likes: ""}, at: -1)
     |> assign(
       current_id: unique_id,
       current_author: name,
       current_content: "",
       current_likes: ""
     )}
  end

  def handle_info({:new_token, token}, socket) do
    if socket.assigns.current_id == nil do
      {:noreply, socket}
    else
      current_author = socket.assigns.current_author
      current_id = socket.assigns.current_id
      current_content = socket.assigns.current_content
      current_likes = socket.assigns.current_likes
      new_content = current_content <> token

      {:noreply,
       socket
       |> stream_insert(
         :posts,
         %{id: current_id, author: current_author, content: new_content, likes: current_likes},
         at: -1
       )
       |> assign(:current_content, new_content)}
    end
  end

  @impl true
  def handle_info({ref, result}, socket) when socket.assigns.task.ref == ref do
    Process.demonitor(ref, [:flush])

    {:noreply, assign(socket, :task, nil)}
  end
end

defmodule PhoenixDemo.Router do
  use Phoenix.Router

  import Phoenix.LiveView.Router

  pipeline :browser do
    plug(:accepts, ["html"])
  end

  scope "/", PhoenixDemo do
    pipe_through(:browser)

    live("/", SampleLive, :index)
  end
end

defmodule PhoenixDemo.Endpoint do
  use Phoenix.Endpoint, otp_app: :sample

  socket("/live", Phoenix.LiveView.Socket)
  plug(PhoenixDemo.Router)
end

defmodule PhoenixDemo.Lorax do
  def inject(model) do
    r = 4
    lora_alpha = 8
    lora_dropout = 0.05

    lora_config = %Lorax.Config{
      r: r,
      alpha: lora_alpha,
      dropout: lora_dropout,
      target_key: true,
      target_query: true,
      target_value: true
    }

    lora_model =
      model
      |> Axon.freeze()
      |> Lorax.inject(lora_config)
  end
end

# Application startup

{:ok, spec} = Bumblebee.load_spec({:hf, "gpt2"})
{:ok, model} = Bumblebee.load_model({:hf, "gpt2"}, spec: spec)
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "gpt2"})
{:ok, generation_config} = Bumblebee.load_generation_config({:hf, "gpt2"})

%{model: model, params: gpt2_params} = model

lora_model = PhoenixDemo.Lorax.inject(model)

lora_serialized =
  Req.get!("https://raw.githubusercontent.com/wtedw/lorax/main/params/elixir-chat-r4a8.lorax").body

lora_only_params = Nx.deserialize(lora_serialized)

merged_params = Map.merge(gpt2_params, lora_only_params)

lora_model_info = %{model: lora_model, params: merged_params, spec: spec}

lora_generation_config =
  Bumblebee.configure(generation_config,
    max_new_tokens: 512,
    strategy: %{type: :multinomial_sampling, top_p: 0.70}
  )

serving =
  Bumblebee.Text.generation(lora_model_info, tokenizer, lora_generation_config,
    compile: [batch_size: 1, sequence_length: 512],
    seed: 1337,
    stream: true,
    defn_options: [compiler: EXLA, lazy_transfers: :always]
  )

{:ok, _} =
  Supervisor.start_link(
    [
      {Nx.Serving, serving: serving, name: PhoenixDemo.Serving, batch_timeout: 100},
      PhoenixDemo.Endpoint
    ],
    strategy: :one_for_one
  )

Process.sleep(:infinity)
