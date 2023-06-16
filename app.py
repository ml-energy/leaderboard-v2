import gradio as gr
import requests
import pandas as pd
from huggingface_hub.hf_api import SpaceInfo
import matplotlib.pyplot as plt
import plotly.express as px

# model_perf_table = "data/test.csv"
# logo_path = "img/image.png"


class TableManager:
    def __init__(self, model_perf_table="data/test.csv"):
        self.model_perf_table = model_perf_table
        self.df = self.get_blocks_party_spaces()

    def get_blocks_party_spaces(self):
        df = pd.read_csv(self.model_perf_table)
        df = df.sort_values(by=['score'], ascending=False)
        return df

    def get_blocks_party_spaces_with_formula(self, formula=None):
        if formula:
            try:
                self.df[str(formula)] = self.df.eval(formula)
                self.generate_dropdown_attr()
            except:
                pass # Handle this error properly in your code
        return self.df

    def get_dropdown(self ):

        self.dropdown = [gr.inputs.Dropdown(choices=self.get_blocks_party_spaces().columns.tolist()[1:], label="X-axis"),
                             gr.inputs.Dropdown(choices=self.get_blocks_party_spaces().columns.tolist()[1:], label="Y-axis"),
                             gr.inputs.Dropdown(choices=[None]+self.get_blocks_party_spaces().columns.tolist()[1:], label="Z-axis (Optional)")]
        return self.dropdown

    def generate_dropdown(self):
        return gr.Dropdown.update(choices=  self.df.columns.tolist()[1:] ), \
               gr.Dropdown.update(choices=  self.df.columns.tolist()[1:] ), \
               gr.Dropdown.update(choices=  self.df.columns.tolist()[1:] )

    def create_scatter(self, x, y, z):
        if z is None or z == 'None' or z == '':
            fig, ax = plt.subplots()

            ax.scatter(list(self.df[x]),list(self.df[y]), marker='o', s=50, c='blue')
            for i, label in enumerate(list(self.df['model'])):
                ax.text(list(self.df[x])[i],list(self.df[y])[i],str(label))
            ax.set_xlabel(x)
            ax.set_ylabel(y)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        else:
            fig = px.scatter_3d(self.df, x=x, y=y, z=z, text=self.df['model'])

            # Set axis labels and title
            fig.update_layout(scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z,
            ),
                title='3D Scatter Plot'
            )
        return fig


def launch():
    table_manager = TableManager()
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
                    data = gr.outputs.Dataframe(type="pandas" )
                with gr.Row():
                    formula_input = gr.inputs.Textbox(lines=1, label="User Designed Column", placeholder = 'e.g. verbosity/latency')
                    data_run = gr.Button("Add To Table")
                    data_run.click(table_manager.get_blocks_party_spaces_with_formula, inputs=formula_input,
                                   outputs=data)

                with gr.Row():
                    with gr.Column():
                        scatter_input = table_manager.get_dropdown()

                        data_run.click(table_manager.generate_dropdown, inputs=None, outputs=scatter_input)
                        fig_run = gr.Button("Generate Figure")

                    with gr.Column():
                        gen_figure = gr.Plot()
                        fig_run.click(table_manager.create_scatter, inputs=scatter_input, outputs=gen_figure)

            with gr.TabItem("About"):
                gr.Markdown(f"""
                            ## Metrics: 
                            - **Human Score**: The average score given by human evaluators.
                            - **Throughput**: The number of tokens generated per second.
                            - **Verbosity**: The average number of generated tokens in the model's response.
                            - **Latency**: The average time it takes for the model to generate a response.
                            - **Memory**: The base memory usage of the model.
                            """)
        # running the function on page load in addition to when the button is clicked
        block.load(table_manager.get_blocks_party_spaces_with_formula, inputs=None, outputs=data)

    # block.launch(share=True)
    block.launch( )


if __name__ == "__main__":
    launch()