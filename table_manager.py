import pandas as pd
import gradio as gr
import plotly.express as px

import json

class TableManager:
    def __init__(self, data_dir: str, setup_list = ['A40_chat', 'A100_chat']) -> None:
        """Load leaderboard data from CSV files in data_dir."""
        # Load and merge CSV files.
        self.data_dir = data_dir
        self.setup_list = setup_list

        self.models = json.load(open(f"{data_dir}/models.json"))
        df = {}
        for setup in setup_list:
            df[setup] = self._read_table(setup)

        self.dfs = df
        self.cur_df_name = next(iter(df.keys()))
        self.default_colname = self.dfs[self.cur_df_name].columns.tolist()[1:]


    def _read_table(self, setup_name: str) -> pd.DataFrame:
        """Read tables."""
        df1 = pd.read_csv(f"{self.data_dir}/score.csv")
        df2 = pd.read_csv(f"{self.data_dir}/{setup_name}_benchmark.csv")
        df = pd.merge(df1, df2, on="model").round(2)

        # Add the #params column.
        df["parameters"] = df["model"].apply(lambda x: self.models[x]["params"])

        # Make the first column (model) a HTML anchor to the model's website.
        def format_model_link(model_name: str) -> str:
            url = self.models[model_name]["url"]
            nickname = self.models[model_name]["nickname"]
            return (
                f'<a style="text-decoration: underline; text-decoration-style: dotted" '
                f'target="_blank" href="{url}">{nickname}</a>'
            )

        df["model"] = df["model"].apply(format_model_link)

        # Sort by energy.
        df = df.sort_values(by="energy", ascending=True)

        return df

    def switch_table(self, setup_name: str) -> None:
        """Switch to a different table."""
        self.cur_df_name = setup_name
        return self.dfs[setup_name]

    def get_df(self):
        """Return the leaderboard Pandas DataFrame."""
        return self.dfs[self.cur_df_name]

    def get_datatypes(self):
        """Return the datatypes of the leaderboard Pandas DataFrame."""
        return ["markdown"] + ["number"] * (len(self.dfs[self.cur_df_name].columns) - 1)
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
            while (column_name := f"custom{counter}") in self.dfs[self.cur_df_name].columns:
                counter += 1

        # If the user did not provide a formula, return an error message.
        if not formula:
            return self.dfs[self.cur_df_name], self._format_msg("Please enter a formula.")

        # If there is an equal sign in the formula, `df.eval` will
        # return an entire DataFrame with the new column, instead of
        # just the new column. This is not what we want, so we check
        # for this case and return an error message.
        if "=" in formula:
            return self.dfs[self.cur_df_name], self._format_msg("Invalid formula: expr cannot contain '='.")

        # The user may want to update an existing column.
        verb = "Updated" if column_name in self.dfs[self.cur_df_name].columns else "Added"

        # Evaluate the formula and catch any error.
        try:
            col = self.dfs[self.cur_df_name].eval(formula)
            if isinstance(col, pd.Series):
                col = col.round(2)
            self.dfs[self.cur_df_name][column_name] = col
        except Exception as exc:
            return self.dfs[self.cur_df_name], self._format_msg(f"Invalid formula: {exc}")

        return [self.dfs[self.cur_df_name],
                self._format_msg(f"{verb} column '{column_name}' in Table {self.cur_df_name}.")] + \
                self.update_dropdown()

    def add_ext_column(self, table_name: str, column_name: str):
        """Add a new column from external table."""

        new_col = f"{table_name}_{column_name}"
        if table_name in self.dfs.keys() and column_name in self.dfs[table_name].columns:
            self.dfs[self.cur_df_name][new_col] = self.dfs[table_name][column_name]
        return [self.dfs[self.cur_df_name]] + self.update_dropdown()

    def get_dropdown(self):
        """Return the dropdown menu."""
        columns = self.dfs[self.cur_df_name].columns.tolist()[1:]
        return [
            gr.Dropdown(choices=columns, label="X"),
            gr.Dropdown(choices=columns, label="Y"),
            gr.Dropdown(choices=columns, label="Z (optional)"),
        ]

    def update_dropdown(self):
        """Update the dropdown menu."""
        columns = self.dfs[self.cur_df_name].columns.tolist()[1:]
        dropdown_update = gr.Dropdown.update(choices=columns)
        return [dropdown_update] * 3

    def get_table_schema(self):
        """Return the schema of table names and column names."""
        return [
            gr.Dropdown(choices=self.dfs.keys(), label="Table Name"),
            gr.Dropdown(choices=self.default_colname, label="Column Name"),
        ]

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
        text = self.dfs[self.cur_df_name]["model"].apply(lambda a: a.split(">")[1].split("<")[0])
        if z is None or z == "None" or z == "":
            fig = px.scatter(self.dfs[self.cur_df_name], x=x, y=y, text=text)
            fig.update_traces(textposition="top center")
        else:
            fig = px.scatter_3d(self.dfs[self.cur_df_name], x=x, y=y, z=z, text=text)
            fig.update_traces(textposition="top center")

        fig.update_layout(width=width, height=height)

        return fig, width, height, ""



