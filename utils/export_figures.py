import json
import math
from itertools import cycle
from typing import Callable, Generator

import torch as T
from utils.data_management import ExperimentDataType, ExperimentReader
from plotly import graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from pathlib import Path

FIGURES_PATH = Path(__file__).parents[1] / "figures"

TEMPLATE = "simple_white"


def plotly_colors() -> Generator[str, None, None]:
    """
    Yields sequences of distinctive colors for plotly plots

    Yields
    ------
    Generator[str, None, None]
        Yields indefinitely a cycle of distinctive colors
    """

    yield from cycle(
        (
            "#FF5733",
            "#33FF57",
            "#3357FF",
            "#FF33A1",
            "#33FFF6",
            "#F3FF33",
            "#9933FF",
            "#FF9633",
            "#33FF99",
            "#FF3333",
        )
    )


def create_metric_graphs(experiment_reader: ExperimentReader) -> dict[str, go.Figure]:
    """
    Plots the graphs of an experiment for all metrics
    """

    # maps a metric -> a dict of model -> values
    graphs: dict[str, dict[str, T.Tensor]] = {}
    num_tasks = -1

    for k, v in experiment_reader.data.items():
        # only process result items that are losses

        # TODO: FIXME
        # scuffed but it works to find num_tasks
        # which should really be bundled up in experiment_writer's config
        if k.endswith("predictions") and isinstance(v, list):
            num_tasks = len(v)

        if not k.endswith("loss"):
            continue

        # grab the model and metric from the result key
        model, metric, _ = k.rsplit("_", 2)


        if metric not in graphs:
            graphs[metric] = {}

        assert isinstance(v, T.Tensor)
        graphs[metric][model] = v

    plots = {}

    # Generate a graph for each metric
    # where each trace is a different model
    for metric, metric_data in graphs.items():
        traces = []

        num_points = -1

        for (model, values), color in zip(metric_data.items(), plotly_colors()):
            trace = go.Scatter(
                y=values,
                name=model.upper().replace("_", " "),
                showlegend=True,
                line=go.scatter.Line(color=color),
            )
            traces.append(trace)
            num_points = len(values)

        plot = go.Figure(
            traces,
            layout=go.Layout(
                title=go.layout.Title(text=f"{metric.capitalize()} Loss"),
                title_x=0.5,
                xaxis_title="Training Batch",
                yaxis_title=f"{metric.capitalize()} Loss (RMSE)",
                titlefont=go.layout.title.Font(color="black"),
                template=TEMPLATE,
            ),
        )
        for i in range(1, num_tasks):
            plot.add_vline(
                x=i * (num_points // num_tasks),
                line_width=3,
                line_dash="dash",
                line_color="dimgray",
            )
        plots[metric] = plot

    return plots


def plot_1d_prediction_graph(experiment_reader: ExperimentReader) -> None:
    """
    Creates prediction graphs for 1d functions, i.e., curves
    """

    # maps a model -> a dict of task -> values
    predictions: dict[str, dict[str, list[T.Tensor] | T.Tensor]] = {}

    for k, v in experiment_reader.data.items():
        if not k.endswith("predictions"):
            continue

        model, metric, _ = k.rsplit("_", 2)

        if model not in predictions:
            predictions[model] = {}

        predictions[model][metric] = v

    assert "base" in predictions

    # get function specific data like num tasks, num points, and graph range
    # from the metric baselines
    num_tasks = None
    num_points = None
    graph_range = None

    for v in predictions["base"].values():
        if isinstance(v, list):
            # num tasks is len of list since predictions are made each task
            num_tasks = len(v)
        else:
            # num points is len of v if it is a tensor because then v represents the baseline for the whole graph
            # as such, finding the min and max of v will let you find the range of the function
            num_points = len(v)
            graph_range = [T.min(v).item() - 0.25, T.max(v).item() + 0.25]

    # make sure all of the function specific data was found from the baseline
    assert num_tasks is not None and num_points is not None and graph_range is not None

    # create subplots where each row is a model and each column is a task
    plot = make_subplots(rows=len(predictions), cols=num_tasks)
    plot.update_xaxes(showticklabels=False)
    plot.update_yaxes(showticklabels=False, range=graph_range)
    plot.update_layout({"title": {"text": "Predictions"}})

    for metric, values in predictions["base"].items():
        # plot all non task specific baselines on all subplots
        if isinstance(values, T.Tensor):
            for row_idx in range(len(predictions)):
                for col_idx in range(num_tasks):
                    plot.add_trace(
                        go.Scatter(
                            x=T.linspace(0, num_tasks, len(values)),
                            y=values.squeeze(),
                            opacity=0.1,
                            line={"color": "lightblue"},
                            name="Base Function",
                            legendgroup="base_background",
                            showlegend=row_idx + col_idx == 0,
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    for row_idx, ((model, task_data), color) in enumerate(
        zip(predictions.items(), plotly_colors())
    ):
        for col_idx in range(num_tasks):
            for metric, values in task_data.items():
                if isinstance(values, list):
                    # if prediction is the same length as the max function length baseline
                    # then plot it over the entire graph
                    # otherwise its a graph of a task and shold be plotted on a subset of the graph
                    if len(values[col_idx]) == num_points:
                        x = T.linspace(0, num_tasks, num_points)
                    else:
                        x = T.linspace(col_idx, col_idx + 1, len(values[col_idx]))

                    y = values[col_idx]
                    plot.add_trace(
                        go.Scatter(
                            x=x.squeeze(),
                            y=y.squeeze(),
                            line={"color": color},
                            name=f"{model.capitalize()} {metric.capitalize()}",
                            legendgroup=model,
                            showlegend=col_idx == 0,
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    st.plotly_chart(plot)


def plot_2d_prediction_graph(experiment_reader: ExperimentReader) -> None:
    """
    Creates prediction graphs for 2d functions, i.e., surfaces
    """

    # maps a model -> a dict of task -> values
    predictions: dict[str, dict[str, list[T.Tensor] | T.Tensor]] = {}

    for k, v in experiment_reader.data.items():
        if not k.endswith("predictions"):
            continue

        model, metric, _ = k.rsplit("_", 2)

        if model not in predictions:
            predictions[model] = {}

        predictions[model][metric] = v

    assert "base" in predictions

    # get function specific data like num tasks, num points, and graph range
    # from the metric baselines
    num_tasks = None
    num_points = None
    graph_range = None

    for v in predictions["base"].values():
        if isinstance(v, list):
            # num tasks is len of list since predictions are made each task
            num_tasks = len(v)
        else:
            # num points is len of v if it is a tensor because then v represents the baseline for the whole graph
            # as such, finding the min and max of v will let you find the range of the function
            num_points = len(v)
            graph_range = [T.min(v).item() - 0.25, T.max(v).item() + 0.25]

    # make sure all of the function specific data was found from the baseline
    assert num_tasks is not None and num_points is not None and graph_range is not None

    # 2d task so num_points represents num_points on each axis
    num_points = round(math.sqrt(num_points))

    # create subplots where each row is a model and each column is a task
    plot = make_subplots(
        rows=len(predictions) - 1,
        cols=num_tasks,
        specs=[
            [{"type": "surface"} for _ in range(num_tasks)]
            for _ in range(len(predictions) - 1)
        ],
    )
    # add plot title
    plot.update_layout({"title": {"text": "Predictions"}})

    # remove ticks from all subplots
    plot.update_layout(
        {
            f"scene{i}": {
                axis: {"showticklabels": False} for axis in ("xaxis", "yaxis", "zaxis")
            }
            for i in range(1, num_tasks * len(predictions))
        }
    )

    for metric, values in predictions["base"].items():
        # plot all non task specific baselines on all subplots
        if isinstance(values, T.Tensor):
            for row_idx in range(len(predictions) - 1):
                for col_idx in range(num_tasks):
                    plot.add_trace(
                        go.Surface(
                            z=values.reshape([round(math.sqrt(num_points))] * 4)
                            .permute(0, 2, 1, 3)
                            .reshape(num_points, num_points),
                            name=f"{model.capitalize()} {metric.capitalize()}",
                            legendgroup=model,
                            showlegend=row_idx + col_idx == 0,
                            opacity=0.1,
                            showscale=False,
                            hoverinfo="skip",
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    # don't draw other baseline stuff for 2d
    del predictions["base"]

    for row_idx, (model, task_data) in enumerate(predictions.items()):
        for col_idx in range(num_tasks):
            for metric, values in task_data.items():
                if isinstance(values, list):
                    plot.add_trace(
                        go.Surface(
                            z=values[col_idx]
                            .reshape([round(math.sqrt(num_points))] * 4)
                            .permute(0, 2, 1, 3)
                            .reshape(num_points, num_points),
                            name=f"{model.capitalize()} {metric.capitalize()}",
                            legendgroup=model,
                            showlegend=col_idx == 0,
                            showscale=False,
                            hoverinfo="skip",
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    st.plotly_chart(plot)


def plot_prediction_graph(experiment_reader: ExperimentReader) -> None:
    """
    Calls either `plot_1d_prediction_graphs` or `plot_2d_prediction_graphs`
    depending on `experiment_reader.experiment_dtype`
    """

    match experiment_reader.experiment_dtype:
        case ExperimentDataType.function_1d:
            plot_1d_prediction_graph(experiment_reader)
        case ExperimentDataType.function_2d:
            plot_2d_prediction_graph(experiment_reader)


def write_data(experiment_reader: ExperimentReader) -> None:
    """
    Writes the data section of the experiment_reader to streamlit
    """

    for name, obj in experiment_reader.data.items():
        if isinstance(obj, dict):
            st.write(f"{name}: { {k for k in obj.keys()} }")
        elif isinstance(obj, list):
            st.write(f"{name}: [{obj[0].shape} (x{len(obj)})]")
        else:
            st.write(f"{name}: {obj.shape}")
        with st.expander("View Data"):
            st.write(str(obj))


def write_config(experiment_reader: ExperimentReader) -> None:
    """
    Writes the config section of the experiment_reader to streamlit
    """

    for name, obj in experiment_reader.config.items():
        st.write(f"{name}: {json.dumps(obj, indent=4)}")


def main() -> None:
    for experiment in ExperimentReader.get_experiments():
        reader = ExperimentReader(experiment)
        reader.read()

        experiment_path = FIGURES_PATH / experiment
        experiment_path.mkdir(exist_ok=True)

        metric_graphs = create_metric_graphs(reader)

        for metric, graph in metric_graphs.items():
            graph_path = experiment_path / f"{metric}_figure.png"
            graph.write_image(graph_path)

        break

        print([*metric_graphs.keys()])


if __name__ == "__main__":
    main()
