import gradio as gr

from helpers import clear_cuda_cache, index_files, bot, user

with gr.Blocks() as demo:
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            files = gr.File(
                file_count="multiple",
                file_types=[".txt", ".pdf", ".doc", ".docx"],
                height=600,
            )
            submit_btn = gr.Button(value="Submit", variant="primary")
            msg_out = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Click 'Submit' to create embeddings",
            )

        submit_btn.click(fn=index_files, inputs=files, outputs=msg_out).then(
            fn=clear_cuda_cache
        )

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(label="Chatbot", height=733)
            with gr.Row():
                input = gr.Textbox(
                    scale=9,
                    placeholder="Type a message...",
                    container=False,
                    autofocus=True,
                    max_lines=3,
                )
                send_btn = gr.Button(scale=1, value="Send", variant="primary")

        input.submit(
            fn=user, inputs=[input, chatbot], outputs=[input, chatbot], queue=False
        ).then(fn=bot, inputs=chatbot, outputs=chatbot).then(fn=clear_cuda_cache)
        send_btn.click(user, [input, chatbot], [input, chatbot], queue=False).then(
            fn=bot, inputs=chatbot, outputs=chatbot
        ).then(fn=clear_cuda_cache)


if __name__ == "__main__":
    demo.queue()
    demo.launch()
