from __future__ import annotations

import os
import json
import yaml
import itertools
import contextlib

import gradio as gr
import pandas as pd
import plotly.io as pio
import plotly.express as px
import numpy as np
pio.templates.default = "plotly_white"


class TableManager:
    def __init__(self, data_dir: str) -> None:
        """Load leaderboard data from CSV files in data_dir."""
        # Load and merge CSV files.
        df = self._read_table(data_dir)
        models = json.load(open(f"{data_dir}/models.json"))

        # Add the #params column.
        df["parameters"] = df["model"].apply(lambda x: models[x]["params"])

        # Make the first column (model) a HTML anchor to the model's website.
        def format_model_link(model_name: str) -> str:
            url = models[model_name]["url"]
            nickname = models[model_name]["nickname"]
            return (
                f'<a style="text-decoration: underline; text-decoration-style: dotted" '
                f'target="_blank" href="{url}">{nickname}</a>'
            )

        df["model"] = df["model"].apply(format_model_link)

        # Sort by energy.
        df = df.sort_values(by="energy", ascending=True)

        self.df = df

    def _read_table(self, data_dir: str) -> pd.DataFrame:
        """Read tables."""
        df_score = pd.read_csv(f"{data_dir}/score.csv")

        with open(f"{data_dir}/schema.yaml", 'r') as file:
            self.schema = yaml.safe_load(file)
            self.cur_filter = list(self.schema.values())

        res_df = pd.DataFrame()

        for choice in itertools.product(*self.schema.values()):
            filepath = f"{data_dir}/{'_'.join(choice)}_benchmark.csv"
            with contextlib.suppress(FileNotFoundError):
                df = pd.read_csv(filepath)
                for key, val in zip(self.schema.keys(), choice):
                    df.insert(1, key, val)
                res_df = pd.concat([res_df, df])

        res_df = pd.merge(df_score, res_df, on=['model']).round(2)
        return res_df

    def get_df(self):
        """Return the leaderboard Pandas DataFrame."""
        return self.df

    def get_datatypes(self):
        """Return the datatypes of the leaderboard Pandas DataFrame."""
        return ["markdown"] + ["number"] * (len(self.df.columns) - 1)
        # return "markdown"

    def _format_msg(self, text: str) -> str:
        """Formats into HTML that prints in Monospace font."""
        return f"<pre style='font-family: monospace'>{text}</pre>"

    def add_column(self, column_name: str, formula: str):
        """Create and add a new column with the given formula."""
        # If the user did not provide the name of the new column,
        # generate a unique name for them.
        if not column_name:
            counter = 1
            while (column_name := f"custom{counter}") in self.df.columns:
                counter += 1

        # If the user did not provide a formula, return an error message.
        if not formula:
            return self.create_filter_df(), self._format_msg("Please enter a formula.")

        # If there is an equal sign in the formula, `df.eval` will
        # return an entire DataFrame with the new column, instead of
        # just the new column. This is not what we want, so we check
        # for this case and return an error message.
        if "=" in formula:
            return self.create_filter_df(), self._format_msg("Invalid formula: expr cannot contain '='.")

        # The user may want to update an existing column.
        verb = "Updated" if column_name in self.df.columns else "Added"

        # Evaluate the formula and catch any error.
        try:
            col = self.df.eval(formula)
            if isinstance(col, pd.Series):
                col = col.round(2)
            self.df[column_name] = col
        except Exception as exc:
            return self.create_filter_df(), self._format_msg(f"Invalid formula: {exc}")
        return self.create_filter_df(), self._format_msg(f"{verb} column '{column_name}'.")

    def get_dropdown(self):
        columns = self.df.columns.tolist()[1:] # include gpu and task in the dropdown
        return [
            gr.Dropdown(choices=columns, label="X"),
            gr.Dropdown(choices=columns, label="Y"),
            gr.Dropdown(choices=columns, label="Z (optional)"),
        ]

    def update_dropdown(self):
        columns = self.df.columns.tolist()[1:]
        dropdown_update = gr.Dropdown.update(choices=columns)
        return [dropdown_update] * 3

    def create_filter_df(self, *filters):
        """Create a filtered dataframe based on the user's choice."""

        self.cur_filter = filters if filters else self.cur_filter
        index = np.full(len(self.df), True)
        for setup, choice in zip(self.schema, self.cur_filter):
            index = index & self.df[setup].isin(choice)
        return self.df.loc[index]

    def plot_scatter(self, width, height, x, y, z):
        # The user did not select either x or y.
        if not x or not y:
            return None, width, height, self._format_msg("Please select both X and Y.")

        # Width and height may be an empty string. Then we set them to 600.
        if not width and not height:
            width, height = "600", "600"
        elif not width:
            width = height
        elif not height:
            height = width
        try:
            width, height = int(width), int(height)
        except ValueError:
            return None, width, height, self._format_msg("Width and height should be positive integers.")

        # Strip the <a> tag from model names.
        text = self.create_filter_df()["model"].apply(lambda x: x.split(">")[1].split("<")[0])
        if z is None or z == "None" or z == "":
            fig = px.scatter(self.create_filter_df(), x=x, y=y, text=text)
            fig.update_traces(textposition="top center")
        else:
            fig = px.scatter_3d(self.create_filter_df(), x=x, y=y, z=z, text=text)
            fig.update_traces(textposition="top center")

        fig.update_layout(width=width, height=height)

        return fig, width, height, ""


# Find the latest version of the CSV files in data/
# and initialize the global TableManager.
latest_date = sorted(os.listdir("data/"))[-1]
global_tbm = TableManager(f"data/{latest_date}")

# Custom JS.
# XXX: This is a hack to make the model names clickable.
#      Ideally, we should set `datatype` in the constructor of `gr.DataFrame` to
#      `["markdown"] + ["number"] * (len(df.columns) - 1)` and format models names
#      as an HTML <a> tag. However, because we also want to dynamically add new
#      columns to the table and Gradio < 4.0 does not support updating `datatype` with
#      `gr.DataFrame.update` yet, we need to manually walk into the DOM and replace
#      the innerHTML of the model name cells with dynamically interpreted HTML.
#      Desired feature tracked at https://github.com/gradio-app/gradio/issues/3732
dataframe_update_js = f"""
function format_model_link() {{
    // Iterate over the cells of the first column of the leaderboard table.
    for (let index = 1; index <= {len(global_tbm.get_df())}; index++) {{
        // Get the cell.
        var cell = document.querySelector(
            `#tab-leaderboard > div > div > div > table > tbody > tr:nth-child(${{index}}) > td:nth-child(1) > div > span`
        );

        // This check exists to make this function idempotent.
        // Multiple changes to the Dataframe component may invoke this function,
        // multiple times to the same HTML table (e.g., adding and sorting cols).
        // Thus, we check whether we already formatted the model names by seeing
        // whether the child of the cell is a text node. If it is not,
        // it means we already parsed it into HTML, so we should just return.
        if (cell.firstChild.nodeType != 3) break;

        // Decode and interpret the innerHTML of the cell as HTML.
        var decoded_string = new DOMParser().parseFromString(cell.innerHTML, "text/html").documentElement.textContent;
        var temp = document.createElement("template");
        temp.innerHTML = decoded_string;
        var model_anchor = temp.content.firstChild;

        // Replace the innerHTML of the cell with the interpreted HTML.
        cell.replaceChildren(model_anchor);
    }}

    // Return all arguments as is.
    return arguments
}}
"""

# Custom CSS.
css = """
/* Make ML.ENERGY look like a clickable logo. */
.text-logo {
    color: #27cb63 !important;
    text-decoration: none !important;
}

/* Make the submit button the same color as the logo. */
.btn-submit {
    background: #27cb63 !important;
    color: white !important;
    border: 0 !important;
}

/* Center the plotly plot inside its container. */
.plotly > div {
    margin: auto !important;
}

/* Limit the width of the first column to 300 px. */
table td:first-child,
table th:first-child {
    max-width: 300px;
    overflow: auto;
    white-space: nowrap;
}
"""

block = gr.Blocks(css=css)
with block:
    tbm = gr.State(global_tbm)  # type: ignore
    gr.HTML("<h1><a href='https://ml.energy' class='text-logo'>ML.ENERGY</a> Leaderboard</h1>")

    with gr.Tabs():
        # Tab 1: Leaderboard.
        with gr.TabItem("Leaderboard"):
            with gr.Row():
                with gr.Box():
                    gr.Markdown("## Select inference setup")
                    checkboxes = []
                    for key, choices in global_tbm.schema.items():
                        # Specifying `value` makes everything checked by default.
                        checkboxes.append(gr.CheckboxGroup(choices=choices, value=choices, label=key))

            # Block 1: Leaderboard table.
            with gr.Row():
                dataframe = gr.Dataframe(type="pandas", elem_id="tab-leaderboard")
                dataframe.change(None, None, None, _js=dataframe_update_js)
                # Table automatically updates when users check or uncheck any checkbox.
                for checkbox in checkboxes:
                    checkbox.change(TableManager.create_filter_df, inputs=[tbm, *checkboxes],
                                    outputs=dataframe)

            # Block 2: Allow users to add new columns.
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        colname_input = gr.Textbox("power", lines=1, label="Custom column name")
                        formula_input = gr.Textbox("energy/latency", lines=1, label="Formula")
                with gr.Column(scale=1):
                    with gr.Row():
                        add_col_btn = gr.Button("Add to table (⏎)", elem_classes=["btn-submit"])
                    with gr.Row():
                        clear_input_btn = gr.Button("Clear")
            with gr.Row():
                add_col_message = gr.HTML("")
            colname_input.submit(TableManager.add_column, inputs=[tbm, colname_input, formula_input],
                                 outputs=[dataframe, add_col_message])
            formula_input.submit(TableManager.add_column, inputs=[tbm, colname_input, formula_input],
                                 outputs=[dataframe, add_col_message])
            add_col_btn.click(TableManager.add_column, inputs=[tbm, colname_input, formula_input],
                              outputs=[dataframe, add_col_message])
            clear_input_btn.click(lambda: (None, None, None), None,
                                  outputs=[colname_input, formula_input, add_col_message])

            # Block 3: Allow users to plot 2D and 3D scatter plots.
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        # Initialize the dropdown choices with the global TableManager with just the original columns.
                        axis_dropdowns = global_tbm.get_dropdown()
                with gr.Column(scale=1):
                    with gr.Row():
                        plot_btn = gr.Button("Plot", elem_classes=["btn-submit"])
                    with gr.Row():
                        clear_plot_btn = gr.Button("Clear")
            with gr.Accordion("Plot size (600 x 600 by default)", open=False):
                with gr.Row():
                    plot_width_input = gr.Textbox("600", lines=1, label="Width (px)")
                    plot_height_input = gr.Textbox("600", lines=1, label="Height (px)")
            with gr.Row():
                plot = gr.Plot()
            with gr.Row():
                plot_message = gr.HTML("")
            add_col_btn.click(TableManager.update_dropdown, inputs=tbm, outputs=axis_dropdowns)  # type: ignore
            plot_width_input.submit(
                TableManager.plot_scatter,
                inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                outputs=[plot, plot_width_input, plot_height_input, plot_message],
            )
            plot_height_input.submit(
                TableManager.plot_scatter,
                inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                outputs=[plot, plot_width_input, plot_height_input, plot_message],
            )
            plot_btn.click(
                TableManager.plot_scatter,
                inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                outputs=[plot, plot_width_input, plot_height_input, plot_message],
            )
            clear_plot_btn.click(
                lambda: (None,) * 7,
                None,
                outputs=[*axis_dropdowns, plot, plot_width_input, plot_height_input, plot_message],
            )

            # Block 4: Leaderboard date.
            with gr.Row():
                gr.HTML(f"<h3 style='color: gray'>Date: {latest_date}</h3>")

        # Tab 2: About page.
        with gr.TabItem("About"):
            # Skip the YAML front matter and title in README.md.
            lines = open("README.md").readlines()
            i = 0
            for i, line in enumerate(lines):
                if line.startswith("# "):
                    i += 2
                    break
            gr.Markdown("\n".join(lines[i:]))

    # Load the table on page load.
    block.load(TableManager.get_df, inputs=tbm, outputs=dataframe)

block.launch()
