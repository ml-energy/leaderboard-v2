from __future__ import annotations

import json
import yaml
import requests
import itertools
import contextlib
import argparse
import os
from typing import Literal
from dateutil import parser, tz

import numpy as np
import gradio as gr
import pandas as pd
import plotly.io as pio
import plotly.express as px
from pandas.api.types import is_numeric_dtype, is_float_dtype
pio.templates.default = "plotly_white"

from spitfight.colosseum.client import ControllerClient


class TableManager:
    def __init__(self, data_dir: str) -> None:
        """Load leaderboard data from CSV files in data_dir.

        Inside `data_dir`, there should be:
        - `models.json`: a JSON file containing information about each model.
        - `schema.yaml`: a YAML file containing the schema of the benchmark.
        - `score.csv`: a CSV file containing the NLP evaluation metrics of each model.
        - `*_benchmark.csv`: CSV files containing the system benchmark results.

        Especially, the `*_benchmark.csv` files should be named after the
        parameters used in the benchmark. For example, for the CSV file that
        contains benchmarking results for A100 and the chat-concise task
        (see `schema.yaml`) for possible choices, the file should be named
        `A100_chat-concise_benchmark.csv`.
        """
        # Load and merge CSV files.
        df = self._read_tables(data_dir)

        # Add the #params column.
        models = json.load(open(f"{data_dir}/models.json"))
        df["parameters"] = df["model"].apply(lambda x: models[x]["params"])

        # Make the first column (model) an HTML anchor to the model's website.
        def format_model_link(model_name: str) -> str:
            url = models[model_name]["url"]
            nickname = models[model_name]["nickname"]
            return (
                f'<a style="text-decoration: underline; text-decoration-style: dotted" '
                f'target="_blank" href="{url}">{nickname}</a>'
            )
        df["model"] = df["model"].apply(format_model_link)

        # Sort by our 'energy efficiency' score.
        df = df.sort_values(by="energy", ascending=True)

        # The full table where all the data are.
        self.full_df = df

        # Default view of the table is to only show the first options.
        self.set_filter_get_df()

    def _read_tables(self, data_dir: str) -> pd.DataFrame:
        """Read tables."""
        df_score = pd.read_csv(f"{data_dir}/score.csv")

        with open(f"{data_dir}/schema.yaml") as file:
            self.schema: dict[str, list] = yaml.safe_load(file)

        res_df = pd.DataFrame()

        # Do a cartesian product of all the choices in the schema
        # and try to read the corresponding CSV files.
        for choice in itertools.product(*self.schema.values()):
            filepath = f"{data_dir}/{'_'.join(choice)}_benchmark.csv"
            with contextlib.suppress(FileNotFoundError):
                df = pd.read_csv(filepath)
                for key, val in zip(self.schema.keys(), choice):
                    df.insert(1, key, val)
                res_df = pd.concat([res_df, df])

        if res_df.empty:
            raise ValueError(f"No benchmark CSV files were read from {data_dir=}.")

        df = pd.merge(res_df, df_score, on=["model"]).round(2)

        # Order columns.
        columns = df.columns.to_list()
        cols_to_order = ["model"]
        cols_to_order.extend(self.schema.keys())
        cols_to_order.append("energy")
        columns = cols_to_order + [col for col in columns if col not in cols_to_order]
        df = df[columns]

        # Delete rows with *any* NaN values.
        df = df.dropna()

        return df

    def _format_msg(self, text: str) -> str:
        """Formats into HTML that prints in Monospace font."""
        return f"<pre style='font-family: monospace'>{text}</pre>"

    def add_column(self, column_name: str, formula: str):
        """Create and add a new column with the given formula."""
        # If the user did not provide the name of the new column,
        # generate a unique name for them.
        if not column_name:
            counter = 1
            while (column_name := f"custom{counter}") in self.full_df.columns:
                counter += 1

        # If the user did not provide a formula, return an error message.
        if not formula:
            return self.cur_df, self._format_msg("Please enter a formula.")

        # If there is an equal sign in the formula, `df.eval` will
        # return an entire DataFrame with the new column, instead of
        # just the new column. This is not what we want, so we check
        # for this case and return an error message.
        if "=" in formula:
            return self.cur_df, self._format_msg("Invalid formula: expr cannot contain '='.")

        # The user may want to update an existing column.
        verb = "Updated" if column_name in self.full_df.columns else "Added"

        # Evaluate the formula and catch any error.
        try:
            # Give the users some helper functions that can be used in the formula
            # like "@sum(response_length)". Also wipe out some global variables.
            col = self.full_df.eval(
                formula,
                local_dict={"sum": sum, "len": len, "max": max, "min": min},
                global_dict={"global_tbm": None},
            )
        except Exception as exc:
            return self.cur_df, self._format_msg(f"Invalid formula: {exc}")

        # If the result is a numeric scalar, make it a Series.
        # We may have deleted some models (rows) form the full dataframe when we
        # called dropna, so we need to query the maximum index instead of taking len.
        if isinstance(col, (int, float)):
            col = pd.Series([col] * (self.full_df.index.max() + 1))
        # We only accept numeric columns.
        if not is_numeric_dtype(col):
            return self.cur_df, self._format_msg("Invalid formula: result must be numeric.")
        # Round if it's floating point.
        if is_float_dtype(col):
            col = col.round(2)

        # If the column already exists, update it.
        if column_name in self.full_df.columns:
            self.full_df[column_name] = col
        else:
            self.full_df.insert(len(self.schema) + 1, column_name, col)

        # If adding a column succeeded, `self.cur_df` should also be updated.
        self.cur_df = self.full_df.loc[self.cur_index]
        return self.cur_df, self._format_msg(f"{verb} column '{column_name}'.")

    def get_dropdown(self):
        columns = self.full_df.columns.tolist()[1:]
        return [
            gr.Dropdown(choices=columns, value="parameters", label="X"),
            gr.Dropdown(choices=columns, value="energy", label="Y"),
            gr.Dropdown(choices=["None", *columns], label="Z (optional)"),
        ]

    def update_dropdown(self):
        columns = self.full_df.columns.tolist()[1:]
        return [
            gr.Dropdown.update(choices=columns),
            gr.Dropdown.update(choices=columns),
            gr.Dropdown.update(choices=["None", *columns]),
        ]

    def set_filter_get_df(self, *filters) -> pd.DataFrame:
        """Set the current set of filters and return the filtered DataFrame."""
        # If the filter is empty, we default to the first choice for each key.
        if not filters:
            filters = [choices[:1] for choices in self.schema.values()]

        index = np.full(len(self.full_df), True)
        for setup, choice in zip(self.schema, filters):
            index = index & self.full_df[setup].isin(choice)
        self.cur_df = self.full_df.loc[index]
        self.cur_index = index
        return self.cur_df

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
        text = self.cur_df["model"].apply(lambda x: x.split(">")[1].split("<")[0])
        # Hide model names since they clutter the plots, and only show them on hover.
        if z is None or z == "None" or z == "":
            fig = px.scatter(self.cur_df, x=x, y=y, hover_name=text)
        else:
            fig = px.scatter_3d(self.cur_df, x=x, y=y, z=z, hover_name=text)
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")))
        fig.update_layout(width=width, height=height)

        return fig, width, height, ""

# The global instance of the TableManager should only be used when
# initializing components in the Gradio interface. If the global instance
# is mutated while handling user sessions, the change will be reflected
# in every user session. Instead, the instance provided by gr.State should
# be used.
global_tbm = TableManager("data")

# Fetch the latest update date of the leaderboard repository.
resp = requests.get("https://api.github.com/repos/ml-energy/leaderboard/commits/master")
if resp.status_code != 200:
    current_date = "[Failed to fetch]"
    print("Failed to fetch the latest release date of the leaderboard repository.")
    print(resp.json())
else:
    current_datetime = parser.parse(resp.json()["commit"]["author"]["date"])
    current_date = current_datetime.astimezone(tz.gettz("US/Eastern")).strftime("%Y-%m-%d")

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
    for (let index = 1; index <= {len(global_tbm.full_df)}; index++) {{
        // Get the cell.
        var cell = document.querySelector(
            `#tab-leaderboard > div > div > div > table > tbody > tr:nth-child(${{index}}) > td:nth-child(1) > div > span`
        );

        // If nothing was found, it likely means that now the visible table has less rows
        // than the full table. This happens when the user filters the table. In this case,
        // we should just return.
        if (cell == null) break;

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
custom_css = """
/* Make ML.ENERGY look like a clickable logo. */
.text-logo {
    color: #23d175 !important;
    text-decoration: none !important;
}

/* Make the submit button the same color as the logo. */
.btn-submit {
    background: #23d175 !important;
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

/* Make tab buttons larger */
.tab-nav > button {
    font-size: 18px !important;
}

/* Color texts. */
.green-text {
    color: #23d175 !important;
}
.red-text {
    color: #ff3860 !important;
}
"""

intro_text = """
<h2>How much energy do modern Large Language Models (LLMs) consume for inference?</h2>

<p style="font-size: 16px">We used <a href="https://ml.energy/zeus">Zeus</a> to benchmark various open source LLMs in terms of how much time and energy they consume for inference.
Time and energy are of course not the only things we care about -- so we also benchmarked all of the models on a variety of NLP datasets,
including the ARC Challenge (reasoning), HellaSwag (common sense), and TruthfulQA (truthfulness).</p>

<p style="font-size: 16px">For more detailed information, please take a look at the <b>About</b> tab.
Every benchmark is limited in some sense -- Before you interpret the results, please take a look at the <b>Limitations</b> section there, too.</p>
"""

# The app will not start without a controller address set.
controller_addr = os.environ["COLOSSEUM_CONTROLLER_ADDR"]
global_controller_client = ControllerClient(controller_addr=controller_addr, timeout=15)

ANONYMOUS_MODEL_TEXT = "## Anonymous 🤫"

# Colosseum helper functions.
def enable_interact():
    return [gr.update(interactive=True)] * 2

def disable_interact():
    return [gr.update(interactive=False)] * 2

def consumed_less_energy_message(energy_a, energy_b):
    """Return a message that indicates that the user chose the model that consumed less energy.

    By default report in "%f %" but if the difference is larger than 2 times, report in "%f X".
    """
    less_energy = min(energy_a, energy_b)
    more_energy = max(energy_a, energy_b)
    factor = less_energy / more_energy
    if factor <= 0.5:
        message = f"<h2>That response also <span class='green-text'>consumed {1/factor:.1f}X less energy</span>!</h2>"
    else:
        message = f"<h2>That response also <span class='green-text'>consumed {100 - factor * 100:.1f}% less energy</span>!</h2>"
    return message

def consumed_more_energy_message(energy_a, energy_b):
    """Return a message that indicates that the user chose the model that consumed more energy.

    By default report in "%f %" but if the difference is larger than 2 times, report in "%f X".
    """
    less_energy = min(energy_a, energy_b)
    more_energy = max(energy_a, energy_b)
    factor = more_energy / less_energy
    if factor >= 2.0:
        message = f"<h2>That response <span class='red-text'>consumed {factor:.1f}x more energy</span>.</h2>"
    else:
        message = f"<h2>That response <span class='red-text'>consumed {factor * 100 - 100:.1f}% more energy</span>.</h2>"
    return message

# Colosseum event handlers
def add_prompt_disable_submit(prompt, history_a, history_b):
    """Add the user's prompt to the two model's history and disable the submit button."""
    client = global_controller_client.fork()
    return [
        gr.Textbox.update(value=" ", interactive=False),
        gr.Button.update(interactive=False),
        history_a + [[prompt, ""]],
        history_b + [[prompt, ""]],
        client,
    ]

def generate_responses(client: ControllerClient, history_a, history_b):
    """Generate responses for the two models."""
    for resp_a, resp_b in itertools.zip_longest(
        client.prompt(prompt=history_a[-1][0], index=0),
        client.prompt(prompt=history_b[-1][0], index=1),
    ):
        if resp_a is not None:
            history_a[-1][1] += resp_a
        if resp_b is not None:
            history_b[-1][1] += resp_b
        yield [history_a, history_b]

def make_resp_vote_func(victory_index: Literal[0, 1]):
    """Return a function that will be called when the user clicks on response preference vote buttons."""
    def resp_vote_func(client: ControllerClient):
        vote_response = client.response_vote(victory_index=victory_index)
        model_name_a, model_name_b = map(lambda n: f"## {n}", vote_response.model_names)
        energy_a, energy_b = vote_response.energy_consumptions
        # User liked the model that also consumed less energy.
        if (victory_index == 0 and energy_a <= energy_b) or (victory_index == 1 and energy_a >= energy_b):
            energy_message = consumed_less_energy_message(energy_a, energy_b)
            return [
                # Disable response vote buttons
                gr.Button.update(interactive=False), gr.Button.update(interactive=False),
                # Reveal model names
                gr.Markdown.update(model_name_a), gr.Markdown.update(model_name_b),
                # Display energy consumption comparison message
                gr.Markdown.update(energy_message, visible=True),
                # Keep energy vote buttons hidden
                gr.Button.update(visible=False, interactive=False), gr.Button.update(visible=False, interactive=False),
                # Enable reset button
                gr.Button.update(visible=True, interactive=True),
            ]
        # User liked the model that consumed more energy.
        else:
            energy_message = consumed_more_energy_message(energy_a, energy_b)
            return [
                # Disable response vote buttons
                gr.Button.update(interactive=False), gr.Button.update(interactive=False),
                # Leave model names hidden
                gr.Markdown.update(ANONYMOUS_MODEL_TEXT), gr.Markdown.update(ANONYMOUS_MODEL_TEXT),
                # Display energy consumption comparison message
                gr.Markdown.update(energy_message, visible=True),
                # Reveal and enable energy vote buttons
                gr.Button.update(visible=True, interactive=True), gr.Button.update(visible=True, interactive=True),
                # Keep the reset button disabled
                gr.Button.update(visible=False, interactive=False),
            ]
    return resp_vote_func

def make_energy_vote_func(is_worth: bool):
    """Return a function that will be called when the user clicks on energy vote buttons."""
    def energy_vote_func(client: ControllerClient, energy_message: str):
        vote_response = client.energy_vote(is_worth=is_worth)
        model_name_a, model_name_b = map(lambda n: f"## {n}", vote_response.model_names)
        return [
            # Reveal model names
            gr.Markdown.update(model_name_a), gr.Markdown.update(model_name_b),
            # Disable energy vote buttons
            gr.Button.update(interactive=False), gr.Button.update(interactive=False),
            # Enable reset button
            gr.Button.update(interactive=True, visible=True),
            # Append to the energy comparison message
            energy_message[:-5] + (" Fair enough.</h2>" if is_worth else " Wasn't worth it.</h2>"),
        ]
    return energy_vote_func

def play_again():
    return [
        # Clear chatbot history
        None, None,
        # Turn on prompt textbox and submit button
        gr.Textbox.update(value="", interactive=True), gr.Button.update(interactive=True),
        # Mask model names
        gr.Markdown.update(ANONYMOUS_MODEL_TEXT),
        gr.Markdown.update(ANONYMOUS_MODEL_TEXT),
        # Hide energy vote buttons and message
        gr.Button.update(visible=False), gr.Button.update(visible=False), gr.Markdown.update(visible=False),
        # Disable reset button
        gr.Button.update(interactive=False, visible=False),
    ]

focus_prompt_input_js = """
function() {
    for (let textarea of document.getElementsByTagName("textarea")) {
        if (textarea.hasAttribute("autofocus")) {
            textarea.focus();
            return;
        }
    }
}
"""

with gr.Blocks(css=custom_css) as block:
    tbm = gr.State(global_tbm)  # type: ignore
    with gr.Box():
        gr.HTML("<h1><a href='https://ml.energy' class='text-logo'>ML.ENERGY</a> Leaderboard</h1>")

    with gr.Tabs():
        # Tab: Colosseum.
        with gr.TabItem("Colosseum ⚔️️"):
            gr.Markdown(open("docs/colosseum_top.md").read())

            with gr.Group():
                with gr.Row():
                    prompt_input = gr.Textbox(
                        show_label=False,
                        placeholder="Type your prompt and press ENTER",
                        autofocus=True,
                        container=False,
                        scale=20,
                        elem_id="prompt-textarea",
                    )
                    prompt_submit_btn = gr.Button(
                        value="⚔️️ Fight!",
                        elem_classes=["btn-submit"],
                        min_width=60,
                        scale=1,
                    )

            with gr.Row():
                masked_model_names = []
                chatbots = []
                resp_vote_btn_list: list[gr.component.Component] = []
                with gr.Column():
                    with gr.Row():
                        masked_model_names.append(gr.Markdown(ANONYMOUS_MODEL_TEXT))
                    with gr.Row():
                        chatbots.append(gr.Chatbot(label="Model A", elem_id="chatbot", height=600))
                    with gr.Row():
                        left_resp_vote_btn = gr.Button(value="👈 Model A is better", interactive=False)
                        resp_vote_btn_list.append(left_resp_vote_btn)

                with gr.Column():
                    with gr.Row():
                        masked_model_names.append(gr.Markdown(ANONYMOUS_MODEL_TEXT))
                    with gr.Row():
                        chatbots.append(gr.Chatbot(label="Model B", elem_id="chatbot", height=600))
                    with gr.Row():
                        right_resp_vote_btn = gr.Button(value="👉 Model B is better", interactive=False)
                        resp_vote_btn_list.append(right_resp_vote_btn)

            with gr.Row():
                energy_comparison_message = gr.HTML(visible=False)

            with gr.Row():
                worth_energy_vote_btn = gr.Button(value="The better response was worth the extra energy.", visible=False)
                notworth_energy_vote_btn = gr.Button(value="Not really worth it.", visible=False)
                energy_vote_btn_list: list[gr.component.Component] = [worth_energy_vote_btn, notworth_energy_vote_btn]

            with gr.Row():
                play_again_btn = gr.Button("Play again!", visible=False)

            gr.Markdown(open("docs/colosseum_bottom.md").read())

            controller_client = gr.State()


            (prompt_input
                .submit(add_prompt_disable_submit, [prompt_input, *chatbots], [prompt_input, prompt_submit_btn, *chatbots, controller_client], queue=False)
                .then(generate_responses, [controller_client, *chatbots], [*chatbots], queue=True)
                .then(enable_interact, None, resp_vote_btn_list, queue=False))
            (prompt_submit_btn
                .click(add_prompt_disable_submit, [prompt_input, *chatbots], [prompt_input, prompt_submit_btn, *chatbots, controller_client], queue=False)
                .then(generate_responses, [controller_client, *chatbots], [*chatbots], queue=True)
                .then(enable_interact, None, resp_vote_btn_list, queue=False))

            left_resp_vote_btn.click(
                make_resp_vote_func(victory_index=0),
                [controller_client],
                [*resp_vote_btn_list, *masked_model_names, energy_comparison_message, *energy_vote_btn_list, play_again_btn],
                queue=False,
            )
            right_resp_vote_btn.click(
                make_resp_vote_func(victory_index=1),
                [controller_client],
                [*resp_vote_btn_list, *masked_model_names, energy_comparison_message, *energy_vote_btn_list, play_again_btn],
                queue=False,
            )

            worth_energy_vote_btn.click(
                make_energy_vote_func(is_worth=True),
                [controller_client, energy_comparison_message],
                [*masked_model_names, *energy_vote_btn_list, play_again_btn, energy_comparison_message],
                queue=False,
            )
            notworth_energy_vote_btn.click(
                make_energy_vote_func(is_worth=False),
                [controller_client, energy_comparison_message],
                [*masked_model_names, *energy_vote_btn_list, play_again_btn, energy_comparison_message],
                queue=False,
            )

            (play_again_btn
                .click(
                    play_again,
                    None,
                    [*chatbots, prompt_input, prompt_submit_btn, *masked_model_names, *energy_vote_btn_list, energy_comparison_message, play_again_btn],
                    queue=False,
                )
                .then(None, _js=focus_prompt_input_js, queue=False))


        # Tab: Leaderboard.
        with gr.Tab("Leaderboard"):
            with gr.Box():
                gr.HTML(intro_text)

            # Block: Checkboxes to select benchmarking parameters.
            with gr.Row():
                with gr.Box():
                    gr.Markdown("### Benchmark results to show")
                    checkboxes: list[gr.CheckboxGroup] = []
                    for key, choices in global_tbm.schema.items():
                        # Specifying `value` makes everything checked by default.
                        checkboxes.append(gr.CheckboxGroup(choices=choices, value=choices[:1], label=key))

            # Block: Leaderboard table.
            with gr.Row():
                dataframe = gr.Dataframe(type="pandas", elem_id="tab-leaderboard")
            # Make sure the models have clickable links.
            dataframe.change(None, None, None, _js=dataframe_update_js, queue=False)
            # Table automatically updates when users check or uncheck any checkbox.
            for checkbox in checkboxes:
                checkbox.change(TableManager.set_filter_get_df, inputs=[tbm, *checkboxes], outputs=dataframe, queue=False)

            # Block: Allow users to add new columns.
            with gr.Box():
                gr.Markdown("### Add custom columns to the table")
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            colname_input = gr.Textbox(lines=1, label="Custom column name")
                            formula_input = gr.Textbox(lines=1, label="Formula (@sum, @len, @max, and @min are supported)")
                    with gr.Column(scale=1):
                        with gr.Row():
                            add_col_btn = gr.Button("Add to table (⏎)", elem_classes=["btn-submit"])
                        with gr.Row():
                            clear_input_btn = gr.Button("Clear")
                with gr.Row():
                    add_col_message = gr.HTML("")
                gr.Examples(
                    examples=[
                        ["power", "energy / latency"],
                        ["token_per_joule", "response_length / energy"],
                        ["verbose", "response_length > @sum(response_length) / @len(response_length)"],
                    ],
                    inputs=[colname_input, formula_input],
                )
                colname_input.submit(
                    TableManager.add_column,
                    inputs=[tbm, colname_input, formula_input],
                    outputs=[dataframe, add_col_message],
                    queue=False,
                )
                formula_input.submit(
                    TableManager.add_column,
                    inputs=[tbm, colname_input, formula_input],
                    outputs=[dataframe, add_col_message],
                    queue=False,
                )
                add_col_btn.click(
                    TableManager.add_column,
                    inputs=[tbm, colname_input, formula_input],
                    outputs=[dataframe, add_col_message],
                    queue=False,
                )
                clear_input_btn.click(
                    lambda: (None, None, None),
                    inputs=None,
                    outputs=[colname_input, formula_input, add_col_message],
                    queue=False,
                )

            # Block: Allow users to plot 2D and 3D scatter plots.
            with gr.Box():
                gr.Markdown("### Scatter plot (Hover over marker to show model name)")
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
                    plot = gr.Plot(value=global_tbm.plot_scatter(
                        plot_width_input.value,
                        plot_height_input.value,
                        x=axis_dropdowns[0].value,
                        y=axis_dropdowns[1].value,
                        z=axis_dropdowns[2].value,
                    )[0])  # type: ignore
                with gr.Row():
                    plot_message = gr.HTML("")
                add_col_btn.click(TableManager.update_dropdown, inputs=tbm, outputs=axis_dropdowns, queue=False)  # type: ignore
                plot_width_input.submit(
                    TableManager.plot_scatter,
                    inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                    outputs=[plot, plot_width_input, plot_height_input, plot_message],
                    queue=False,
                )
                plot_height_input.submit(
                    TableManager.plot_scatter,
                    inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                    outputs=[plot, plot_width_input, plot_height_input, plot_message],
                    queue=False,
                )
                plot_btn.click(
                    TableManager.plot_scatter,
                    inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                    outputs=[plot, plot_width_input, plot_height_input, plot_message],
                    queue=False,
                )
                clear_plot_btn.click(
                    lambda: (None,) * 7,
                    None,
                    outputs=[*axis_dropdowns, plot, plot_width_input, plot_height_input, plot_message],
                    queue=False,
                )

            # Block: Leaderboard date.
            with gr.Row():
                gr.HTML(f"<h3 style='color: gray'>Last updated: {current_date}</h3>")

        # Tab: About page.
        with gr.Tab("About"):
            # Read in LEADERBOARD.md
            gr.Markdown(open("docs/leaderboard.md").read())

    # Load the table on page load.
    block.load(lambda: global_tbm.set_filter_get_df(), outputs=dataframe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Specify if sharing is enabled")
    parser.add_argument("--concurrency", type=int, default=10)

    args = parser.parse_args()
    block.queue(
        concurrency_count=args.concurrency, status_update_rate=10, api_open=False
    ).launch(share=args.share, show_error=True)
