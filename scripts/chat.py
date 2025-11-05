# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio",
#     "openai",
# ]
# ///
import argparse

import gradio as gr
from openai import OpenAI


def chat_function(message, history, endpoint, model_name, key, temperature, top_p, max_tokens):
    """
    Chat function that communicates with OpenAI-compatible endpoint.
    """
    client = OpenAI(api_key=key, base_url=f"http://{endpoint}/v1" if not endpoint.startswith("http") else endpoint)

    messages = []

    # Convert history to messages format
    for h in history:
        if isinstance(h, (list, tuple)) and len(h) == 2:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})

    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(max_tokens),
        )

        # Simple streaming - just accumulate and return text
        response_text = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                response_text += content
                yield response_text

    except Exception as e:
        yield f"Error: {str(e)}\n\nPlease check your endpoint and model configuration."


def create_demo():
    """Create and configure the Gradio interface."""

    with gr.Blocks(title="LLM Chat Interface") as demo:
        gr.Markdown("# LLM Chat Interface")
        gr.Markdown("Connect to any OpenAI-compatible LLM endpoint")

        with gr.Row():
            with gr.Column(scale=1):
                endpoint_input = gr.Textbox(
                    label="API Endpoint",
                    value="0.0.0.0:8000",
                    placeholder="Enter endpoint (e.g., localhost:8000)",
                    info="URL or address of the OpenAI-compatible API",
                )
                model_input = gr.Textbox(
                    label="Model Name",
                    value="Qwen/Qwen3-8B",
                    placeholder="Enter model name",
                    info="Name of the model to use",
                )
                key_input = gr.Textbox(
                    label="API Key",
                    value="EMPTY",
                    placeholder="Enter API key",
                    info="API key for the OpenAI-compatible API",
                )

                gr.Markdown("### Generation Parameters")
                temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top P")
                max_tokens_slider = gr.Slider(minimum=16, maximum=8192, value=2048, step=16, label="Max Tokens")

            with gr.Column(scale=3):
                # Create chat interface with custom settings
                chat_interface = gr.ChatInterface(  # noqa: F841
                    fn=chat_function,
                    additional_inputs=[
                        endpoint_input,
                        model_input,
                        key_input,
                        temp_slider,
                        top_p_slider,
                        max_tokens_slider,
                    ],
                    chatbot=gr.Chatbot(
                        height=500,
                    ),
                    textbox=gr.Textbox(placeholder="Type your message here..."),
                )

    return demo


def main():
    """Main function to launch the Gradio app."""
    parser = argparse.ArgumentParser(description="LLM Chat UI with OpenAI-compatible endpoint")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument(
        "--no-share", action="store_true", help="Disable public shareable link (share is enabled by default)"
    )
    args = parser.parse_args()

    # Create the demo
    demo = create_demo()

    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=args.port,
        share=not args.no_share,  # Share by default unless --no-share is specified
        show_api=False,
    )


if __name__ == "__main__":
    main()
