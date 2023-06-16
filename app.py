import gradio as gr
import requests
import pandas as pd
from huggingface_hub.hf_api import SpaceInfo
import matplotlib.pyplot as plt
import plotly.express as px

model_perf_table = "data/test.csv"
logo_path = "img/image.png"


def get_blocks_party_spaces():
    df = pd.read_csv(model_perf_table)
    df = df.sort_values(by=['score'],ascending=False)
    return df


def get_blocks_party_spaces_with_formula(formula=None):
    # get the dataframe
    df = get_blocks_party_spaces()
    if formula:
        try:
            df[str(formula)] = df.eval(formula)
        except:
            pass # Handle this error properly in your code
    return df

def create_scatter(x, y, z):
    df = get_blocks_party_spaces()
    if z is None or z == 'None' or z == '':
        fig, ax = plt.subplots()

        ax.scatter(list(df[x]),list(df[y]), marker='o', s=50, c='blue')
        for i, label in enumerate(list(df['model'])):
            ax.text(list(df[x])[i],list(df[y])[i],str(label))
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    else:
        fig = px.scatter_3d(df, x=x, y=y, z=z, text=df['model'])

        # Set axis labels and title
        fig.update_layout(scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
        ),
            title='3D Scatter Plot'
        )

    return fig

block = gr.Blocks()
with block:
    # gr.outputs.HTML(f'<img src="{logo_path}" alt="logo" height="1000px">')
    # img = gr.Image(logo_path,shape=[1,2]).style( rounded=False)

    gr.Markdown(f"""
                # 🦙💦SpitFight - Leaderboard for LLM 
                """)
    with gr.Tabs():
        with gr.TabItem("Leaderboard"):
            with gr.Row():
                data = gr.outputs.Dataframe(type="pandas")
            with gr.Row():
                formula_input = gr.inputs.Textbox(lines=1, label="User Designed Column", placeholder = 'e.g. verbosity/latency')
                data_run = gr.Button("Add To Table")
                data_run.click(get_blocks_party_spaces_with_formula, inputs=formula_input, outputs=data)
            # running the function on page load in addition to when the button is clicked
            with gr.Row():
                with gr.Column():
                    scatter_input = [gr.inputs.Dropdown(choices=get_blocks_party_spaces().columns.tolist()[1:], label="X-axis"),
                         gr.inputs.Dropdown(choices=get_blocks_party_spaces().columns.tolist()[1:], label="Y-axis"),
                         gr.inputs.Dropdown(choices=[None]+get_blocks_party_spaces().columns.tolist()[1:], label="Z-axis (Optional)")]
                    fig_run = gr.Button("Generate Figure")

                with gr.Column():
                    gen_figure = gr.Plot()# gr.outputs.Image(type="pil")
                    fig_run.click(create_scatter, inputs=scatter_input, outputs=gen_figure)


        with gr.TabItem("About"):
            gr.Markdown(f"""
                        ## Metrics: 
                        - **Human Score**: The average score given by human evaluators.
                        - **Throughput**: The number of tokens generated per second.
                        - **Verbosity**: The average number of generated tokens in the model's response.
                        - **Latency**: The average time it takes for the model to generate a response.
                        - **Memory**: The base memory usage of the model.
                        """)

    block.load(get_blocks_party_spaces_with_formula, inputs=None, outputs=data)

block.launch(share=True)
# block.launch( )
