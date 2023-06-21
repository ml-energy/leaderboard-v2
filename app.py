from __future__ import annotations

import os

import gradio as gr
import plotly.io as pio
from table_manager import TableManager
pio.templates.default = "plotly_white"


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
            # Block 0: Choose a table to display
            with gr.Row():
                setup_radio = gr.inputs.Radio(global_tbm.setup_list, default = 'A40_chat', label="Choose GPU Type - Prompt Setup")

            # Block 1: Leaderboard table.
            with gr.Row():
                dataframe = gr.Dataframe(type="pandas", elem_id="tab-leaderboard")
                dataframe.change(None, None, None, _js=dataframe_update_js)
            setup_radio.change(fn = global_tbm.switch_table, inputs = setup_radio, outputs = dataframe)

            # Block 2: Allow users to new columns.
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        colname_input = gr.Textbox("power", lines=1, label="Custom column name")
                        formula_input = gr.Textbox("energy/latency", lines=1, label="Formula")
                with gr.Column(scale=1):
                    with gr.Row():
                        add_col_btn = gr.Button("Add custom column to table (⏎)", elem_classes=["btn-submit"])
                    with gr.Row():
                        clear_input_btn = gr.Button("Clear")
            with gr.Row():
                add_col_message = gr.HTML("")
            # colname_input.submit(global_tbm.add_column, inputs=[tbm, colname_input, formula_input], outputs=[dataframe, add_col_message])
            # formula_input.submit(global_tbm.add_column, inputs=[tbm, colname_input, formula_input], outputs=[dataframe, add_col_message])
            clear_input_btn.click(lambda: (None, None, None), None, outputs=[colname_input, formula_input, add_col_message])

            # Block 3: Allow users to add columns from the dropdown.

            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        add_col_from_other_table = global_tbm.get_table_schema()
                with gr.Column(scale=1):
                    with gr.Row():
                        add_col_from_other_table_btn = gr.Button("Add external column (⏎)", elem_classes=["btn-submit"])

            # add_col_from_other_table_btn.click(global_tbm.add_ext_column, inputs=add_col_from_other_table, outputs=dataframe)

            # Block 4: Allow users to plot 2D and 3D scatter plots.
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

            add_col_btn.click(global_tbm.add_column, inputs=[colname_input, formula_input],
                              outputs=[dataframe, add_col_message] + axis_dropdowns)
            add_col_from_other_table_btn.click(global_tbm.add_ext_column, inputs=add_col_from_other_table,
                                               outputs=[dataframe] + axis_dropdowns)

            plot_width_input.submit(
                global_tbm.plot_scatter,
                inputs=[ plot_width_input, plot_height_input, *axis_dropdowns],
                outputs=[plot, plot_width_input, plot_height_input, plot_message],
            )
            plot_height_input.submit(
                global_tbm.plot_scatter,
                inputs=[  plot_width_input, plot_height_input, *axis_dropdowns],
                outputs=[plot, plot_width_input, plot_height_input, plot_message],
            )
            plot_btn.click(
                global_tbm.plot_scatter,
                inputs=[  plot_width_input, plot_height_input, *axis_dropdowns],
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

