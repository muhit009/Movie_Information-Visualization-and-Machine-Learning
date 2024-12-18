from dash import Dash, dcc, html, Input, Output, State
from dash import dash_table
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from scipy.stats import shapiro, kstest, normaltest, probplot
import io
import base64
import dash
from io import BytesIO
import matplotlib.pyplot as plt

data = pd.read_csv('cleaned_data.csv')
balanced_data = data.copy()
cleaned_data = data.copy()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
numerical_data = data[numerical_columns].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)
global_data = None

app = Dash(__name__)

# Tab 1 Layout
tab1_layout = html.Div([
    html.H1("Cleaned Dataset Viewer and Downloader"),
    html.Div([
        html.H2("Cleaned Data Preview"),
        html.Div(id="data-table", children=[
            dcc.Markdown("Loading data...")
        ]),
    ]),
    html.Div([
        html.H2("Download Cleaned Data"),
        html.A(
            "Download CSV",
            id="download-link",
            download="cleaned_data.csv",
            href="",
            target="_blank",
            style={"display": "block", "marginTop": "20px"}
        )
    ])
])

# Tab 1 Callbacks
@app.callback(
    Output("data-table", "children"),
    Input("data-table", "id")
)
def update_table(_):
    preview = balanced_data.head(10).to_markdown()
    return dcc.Markdown(f"```\n{preview}\n```")

@app.callback(
    Output("download-link", "href"),
    Input("download-link", "id")
)
def generate_download_link(_):
    csv_string = balanced_data.to_csv(index=False)
    csv_encoded = base64.b64encode(csv_string.encode()).decode()
    return f"data:text/csv;base64,{csv_encoded}"

# Tab 2 Layout
tab2_layout = html.Div([
    html.H1("Interactive Data Cleaning Tool"),

    html.Div([
        html.H3("Drop Duplicates"),
        html.Button("Drop Duplicates", id="drop-duplicates-btn", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.H3("Drop Null Values"),
        dcc.Checklist(
            id="drop-null-options",
            options=[
                {'label': 'Drop Rows with Nulls', 'value': 'rows'},
                {'label': 'Drop Columns with Nulls', 'value': 'columns'},
            ],
            value=[]
        ),
        html.Button("Apply", id="drop-null-btn", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.H3("Drop Columns"),
        dcc.Dropdown(
            id="columns-dropdown",
            options=[{"label": col, "value": col} for col in data.columns],
            multi=True,
            placeholder="Select columns to drop"
        ),
        html.Button("Drop Columns", id="drop-columns-btn", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    html.H3("Data Preview"),
    dash_table.DataTable(
        id="cleaning-data-table",
        data=cleaned_data.head(10).to_dict('records'),
        columns=[{"name": i, "id": i} for i in cleaned_data.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    )
])

# Tab 2 Callbacks
@app.callback(
    [Output("cleaning-data-table", "data"),
     Output("cleaning-data-table", "columns"),
     Output("columns-dropdown", "options")],
    Input("drop-duplicates-btn", "n_clicks"),
    Input("drop-null-btn", "n_clicks"),
    Input("drop-null-options", "value"),
    Input("drop-columns-btn", "n_clicks"),
    State("columns-dropdown", "value"),
    prevent_initial_call=True
)
def clean_data(drop_duplicates_clicks, drop_null_clicks, null_options, drop_columns_clicks, columns_to_drop):
    global cleaned_data
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id']

    if trigger == "drop-duplicates-btn.n_clicks":
        cleaned_data = cleaned_data.drop_duplicates()
    elif trigger == "drop-null-btn.n_clicks":
        if 'rows' in null_options:
            cleaned_data = cleaned_data.dropna()
        if 'columns' in null_options:
            cleaned_data = cleaned_data.dropna(axis=1)
    elif trigger == "drop-columns-btn.n_clicks" and columns_to_drop:
        cleaned_data = cleaned_data.drop(columns=columns_to_drop)

    updated_columns = [{"label": col, "value": col} for col in cleaned_data.columns]

    return (
        cleaned_data.head(10).to_dict('records'),
        [{"name": i, "id": i} for i in cleaned_data.columns],
        updated_columns
    )

# Main Layout
# app.layout = html.Div([
#     dcc.Tabs(id="tabs", value='Data', children=[
#         dcc.Tab(label='Data', value='Data'),
#         dcc.Tab(label='Cleaning', value='Cleaning'),
#     ]),
#     html.Div(id='tab-content')
# ])
tab3_layout = html.Div([
    html.H1("Outlier Detection and Removal"),
    html.Div([
        html.H3("Select Numerical Column"),
        dcc.Dropdown(
            id="column-dropdown",
            options=[{"label": col, "value": col} for col in numerical_columns],
            placeholder="Select a numerical column",
        )
    ], style={"marginBottom": "20px"}),
    html.Div(id="plots-container"),
    html.Div([
        html.H3("Select Outlier Detection Method"),
        dcc.RadioItems(
            id="method-radio",
            options=[
                {"label": "Z-Score", "value": "zscore"},
                {"label": "IQR", "value": "iqr"}
            ],
            value="zscore"
        ),
        html.Button("Remove Outliers", id="remove-outliers-btn", n_clicks=0),
    ], style={"marginTop": "20px"}),
    html.Div(id="updated-plots-container")
])

@app.callback(
    Output("plots-container", "children"),
    Input("column-dropdown", "value"),
    prevent_initial_call=True
)
def show_initial_plots(column):
    if column is None:
        return None

    boxplot = px.box(data, y=column, title=f"Boxplot for {column}")
    violinplot = px.violin(data, y=column, title=f"Violin Plot for {column}")

    return html.Div([
        dcc.Graph(figure=boxplot),
        dcc.Graph(figure=violinplot)
    ])

@app.callback(
    Output("updated-plots-container", "children"),
    Input("remove-outliers-btn", "n_clicks"),
    State("column-dropdown", "value"),
    State("method-radio", "value"),
    prevent_initial_call=True
)
def remove_outliers(n_clicks, column, method):
    global data

    if column is None or method is None:
        return None

    column_data = data[column]

    if method == "zscore":
        z_scores = (column_data - column_data.mean()) / column_data.std()
        data = data[np.abs(z_scores) <= 3]
    elif method == "iqr":
        Q1 = column_data.quantile(0.25)
        Q3 = column_data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[(column_data >= Q1 - 1.5 * IQR) & (column_data <= Q3 + 1.5 * IQR)]

    updated_boxplot = px.box(data, y=column, title=f"Updated Boxplot for {column} (Outliers Removed)")
    updated_violinplot = px.violin(data, y=column, title=f"Updated Violin Plot for {column} (Outliers Removed)")

    return html.Div([
        dcc.Graph(figure=updated_boxplot),
        dcc.Graph(figure=updated_violinplot)
    ])

tab4_layout = html.Div([
    html.H1("PCA Visualization Tool"),
    html.Div([
        html.H3("Select Number of Principal Components"),
        dcc.Slider(
            id="pca-slider",
            min=1,
            max=min(len(numerical_columns), 10),
            step=1,
            value=2,
            marks={i: f"{i}" for i in range(1, min(len(numerical_columns), 10) + 1)}
        )
    ], style={"marginBottom": "20px"}),
    html.Div(id="pca-plot-container")
])

@app.callback(
    Output("pca-plot-container", "children"),
    Input("pca-slider", "value"),
    prevent_initial_call=True
)
def perform_pca(n_components):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i + 1}" for i in range(n_components)])
    pca_df["index"] = range(len(pca_df))

    explained_variance = pca.explained_variance_ratio_.cumsum()

    scatter_plot = None
    if n_components >= 2:
        scatter_plot = px.scatter(
            pca_df, x="PC1", y="PC2",
            title=f"PCA Scatter Plot (PC1 vs PC2)",
            labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
        )

    variance_plot = px.line(
        x=range(1, len(explained_variance) + 1),
        y=explained_variance,
        title="Cumulative Explained Variance",
        labels={"x": "Number of Components", "y": "Cumulative Variance"}
    )

    plots = [dcc.Graph(figure=variance_plot)]
    if scatter_plot:
        plots.append(dcc.Graph(figure=scatter_plot))

    return html.Div(plots)

tab5_layout = html.Div([
    html.H1("Normality Testing and Q-Q Plot"),
    html.Div([
        html.H3("Select a Numerical Column"),
        dcc.Dropdown(
            id="normality-column-dropdown",
            options=[{"label": col, "value": col} for col in numerical_columns],
            placeholder="Select a column"
        )
    ], style={"marginBottom": "20px"}),
    html.Div([
        html.H3("Select Normality Test(s)"),
        dcc.Checklist(
            id="normality-test-checklist",
            options=[
                {"label": "Kolmogorov-Smirnov Test", "value": "ks"},
                {"label": "Shapiro-Wilk Test", "value": "shapiro"},
                {"label": "D’Agostino K-Squared Test", "value": "dagostino"}
            ],
            value=["ks", "shapiro", "dagostino"]
        )
    ], style={"marginBottom": "20px"}),
    html.Button("Run Tests", id="run-tests-btn", n_clicks=0, style={"marginBottom": "20px"}),
    html.Div(id="test-results-container"),
    html.H3("Q-Q Plot"),
    dcc.Graph(id="qq-plot")
])

@app.callback(
    [Output("test-results-container", "children"),
     Output("qq-plot", "figure")],
    Input("run-tests-btn", "n_clicks"),
    State("normality-column-dropdown", "value"),
    State("normality-test-checklist", "value"),
    prevent_initial_call=True
)
def run_normality_tests(n_clicks, column, selected_tests):
    if column is None:
        return html.Div("Please select a column to perform tests."), go.Figure()

    x = data[column].dropna()
    if x.empty:
        return html.Div(f"No valid data in the column {column}."), go.Figure()

    results = []
    for test in selected_tests:
        if test == "ks":
            stats, p = kstest(x, 'norm', args=(np.mean(x), np.std(x)))
            results.append(f"K-S Test: Statistics={stats:.2f}, p-value={p:.2f}")
            results.append(f"Result: {'Normal' if p > 0.01 else 'Not Normal'}")
        elif test == "shapiro":
            stats, p = shapiro(x)
            results.append(f"Shapiro Test: Statistics={stats:.2f}, p-value={p:.2f}")
            results.append(f"Result: {'Normal' if p > 0.01 else 'Not Normal'}")
        elif test == "dagostino":
            stats, p = normaltest(x)
            results.append(f"DA K-Squared Test: Statistics={stats:.2f}, p-value={p:.2f}")
            results.append(f"Result: {'Normal' if p > 0.01 else 'Not Normal'}")

    qq_plot = probplot(x, dist="norm", plot=None)
    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(x=qq_plot[0][0], y=qq_plot[0][1], mode='markers', name='Data'))
    qq_fig.add_trace(go.Scatter(x=qq_plot[0][0], y=qq_plot[0][0], mode='lines', name='Ideal Line', line=dict(color='red', dash='dash')))
    qq_fig.update_layout(title=f"Q-Q Plot for {column}", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")

    return html.Div([html.P(res) for res in results]), qq_fig

# Tab 6 Layout
tab6_layout = html.Div([
    html.H1("Data Transformation Tool"),
    html.Div([
        html.H3("Select a Numerical Column"),
        dcc.Dropdown(
            id="transformation-column-dropdown",
            options=[{"label": col, "value": col} for col in numerical_columns],
            placeholder="Select a numerical column"
        )
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.H3("Select a Transformation"),
        dcc.RadioItems(
            id="transformation-radio",
            options=[
                {"label": "Log Transformation", "value": "log"},
                {"label": "Square Root Transformation", "value": "sqrt"},
                {"label": "Reciprocal Transformation", "value": "reciprocal"},
                {"label": "Exponential Transformation", "value": "exp"},
                {"label": "Standard Scaling", "value": "standard"},
                {"label": "Min-Max Scaling", "value": "minmax"}
            ],
            value="log"
        ),
        html.Button("Apply Transformation", id="apply-transformation-btn", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    html.Div(id="transformation-results-container"),

    html.H3("Visualization of Transformed Data"),
    dcc.Graph(id="transformation-plot")
])

@app.callback(
    [Output("transformation-results-container", "children"),
     Output("transformation-plot", "figure")],
    Input("apply-transformation-btn", "n_clicks"),
    State("transformation-column-dropdown", "value"),
    State("transformation-radio", "value"),
    prevent_initial_call=True
)
def apply_transformation(n_clicks, column, transformation):
    if column is None:
        return "Please select a column to transform.", go.Figure()

    x = data[column].dropna()

    try:
        if transformation == "log":
            transformed_data = np.log1p(x)
        elif transformation == "sqrt":
            transformed_data = np.sqrt(x)
        elif transformation == "reciprocal":
            transformed_data = 1 / (x + 1e-8)
        elif transformation == "exp":
            transformed_data = np.exp(x)
        elif transformation == "standard":
            scaler = StandardScaler()
            transformed_data = scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
        elif transformation == "minmax":
            scaler = MinMaxScaler()
            transformed_data = scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
        else:
            return "Invalid transformation selected.", go.Figure()
    except Exception as e:
        return f"Error applying transformation: {e}", go.Figure()

    results = f"Applied {transformation.capitalize()} Transformation to {column}."

    fig = px.histogram(
        x=transformed_data,
        nbins=50,
        title=f"Histogram of {column} (Transformed with {transformation.capitalize()})"
    )
    fig.update_layout(
        xaxis_title="Transformed Values",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    return results, fig

# Tab 7 Layout
tab7_layout = html.Div([
    html.H1("Dynamic Plotting Tool for Numerical Features"),
    html.Div([
        html.H3("Upload a CSV File"),
        dcc.Upload(
            id="upload-data",
            children=html.Button("Upload File"),
            multiple=False
        )
    ], style={"marginBottom": "20px"}),

    html.Div(id="data-preview"),

    html.Div([
        html.H3("Select Plot Type"),
        dcc.Dropdown(
            id="plot-dropdown",
            options=[
                {"label": "Line Plot", "value": "line"},
                {"label": "Bar Plot (Stacked)", "value": "bar_stack"},
                {"label": "Bar Plot (Grouped)", "value": "bar_group"},
                {"label": "Count Plot", "value": "count"},
                {"label": "Pie Chart", "value": "pie"},
                {"label": "Dist Plot", "value": "dist"},
                {"label": "Pair Plot", "value": "pair"},
                {"label": "Heatmap with Color Bar", "value": "heatmap"},
                {"label": "Histogram with KDE", "value": "hist_kde"},
                {"label": "Q-Q Plot", "value": "qq"},
                {"label": "KDE Plot", "value": "kde"},
            ],
            placeholder="Select plot type"
        )
    ], style={"marginBottom": "20px"}),

    html.Div(id="feature-count-info"),

    html.Div([
        html.H3("Select Feature(s)"),
        dcc.Dropdown(
            id="feature-dropdown",
            multi=True,
            placeholder="Select the required features"
        )
    ], style={"marginBottom": "20px"}),

    html.Button("Generate Plot", id="generate-plot-btn", n_clicks=0),

    html.Div(id="plot-container")
])

@app.callback(
    [Output("data-preview", "children"),
     Output("feature-dropdown", "options")],
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def load_data(contents, filename):
    global global_data
    if contents is None:
        return "No file uploaded", []

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith(".csv"):
            global_data = pd.read_csv(BytesIO(decoded))
        elif filename.endswith(".xlsx"):
            global_data = pd.read_excel(BytesIO(decoded))
        else:
            return "Unsupported file type", []

        numerical_columns = global_data.select_dtypes(include=[np.number]).columns
        options = [{"label": col, "value": col} for col in numerical_columns]

        preview = f"Loaded dataset with {global_data.shape[0]} rows and {global_data.shape[1]} columns."

        return preview, options
    except Exception as e:
        return f"Error loading file: {str(e)}", []

@app.callback(
    Output("feature-count-info", "children"),
    Input("plot-dropdown", "value"),
    prevent_initial_call=True
)
def update_feature_count(plot_type):
    if plot_type is None:
        return "Please select a plot type."

    feature_requirements = {
        "line": "This plot requires at least 2 features: one for the x-axis and one or more for the y-axis.",
        "bar_stack": "This plot requires at least 2 features: one for the x-axis and one or more for the y-axis.",
        "bar_group": "This plot requires at least 2 features: one for the x-axis and one or more for the y-axis.",
        "count": "This plot requires 1 feature.",
        "pie": "This plot requires 2 features: one for categories and one for values.",
        "dist": "This plot requires 1 feature.",
        "pair": "This plot requires multiple features.",
        "heatmap": "This plot does not require specific features (it uses correlations).",
        "hist_kde": "This plot requires 1 feature.",
        "qq": "This plot requires 1 feature.",
        "kde": "This plot requires 1 feature."
    }

    return feature_requirements.get(plot_type, "Unknown plot type selected.")

@app.callback(
    Output("plot-container", "children"),
    Input("generate-plot-btn", "n_clicks"),
    State("plot-dropdown", "value"),
    State("feature-dropdown", "value"),
    prevent_initial_call=True
)
def generate_plot(n_clicks, plot_type, features):
    global global_data

    if global_data is None or plot_type is None:
        return "Please upload a dataset and select a plot type."

    try:
        if plot_type in ["line", "bar_stack", "bar_group"]:
            if len(features) < 2:
                return "This plot requires at least 2 features."
            fig = px.line(global_data, x=features[0], y=features[1:]) if plot_type == "line" else \
                  px.bar(global_data, x=features[0], y=features[1:], barmode="stack" if plot_type == "bar_stack" else "group")
        elif plot_type == "count":
            if len(features) != 1:
                return "This plot requires exactly 1 feature."
            fig = px.histogram(global_data, x=features[0])
        elif plot_type == "pie":
            if len(features) != 2:
                return "This plot requires exactly 2 features."
            fig = px.pie(global_data, names=features[0], values=features[1])
        elif plot_type == "dist":
            if len(features) != 1:
                return "This plot requires exactly 1 feature."
            fig = px.histogram(global_data, x=features[0], marginal="box", opacity=0.7)
        elif plot_type == "pair":
            if len(features) < 2:
                return "This plot requires multiple features."
            sns.pairplot(global_data[features])
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            encoded_image = base64.b64encode(buf.read()).decode()
            return html.Img(src=f"data:image/png;base64,{encoded_image}")
        elif plot_type == "heatmap":
            fig = px.imshow(global_data.corr(), color_continuous_scale="viridis")
        elif plot_type == "hist_kde":
            if len(features) != 1:
                return "This plot requires exactly 1 feature."
            fig = px.histogram(global_data, x=features[0], marginal="kde", opacity=0.7)
        elif plot_type == "qq":
            if len(features) != 1:
                return "This plot requires exactly 1 feature."
            sns.set(style="whitegrid")
            sns.histplot(global_data[features[0]], kde=True)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            encoded_image = base64.b64encode(buf.read()).decode()
            return html.Img(src=f"data:image/png;base64,{encoded_image}")
        elif plot_type == "kde":
            if len(features) != 1:
                return "This plot requires exactly 1 feature."
            sns.kdeplot(data=global_data[features[0]], fill=True, alpha=0.6)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            encoded_image = base64.b64encode(buf.read()).decode()
            return html.Img(src=f"data:image/png;base64,{encoded_image}")
        else:
            return "Plot type not implemented yet."

        return dcc.Graph(figure=fig)

    except Exception as e:
        return f"Error generating plot: {str(e)}"


# Tab 8 Layout
tab8_layout = html.Div([
    html.H1("Categorical Data Visualization Tool"),
    html.Div([
        html.H3("Upload a CSV File"),
        dcc.Upload(
            id="upload-data-tab8",
            children=html.Button("Upload File"),
            multiple=False
        )
    ], style={"marginBottom": "20px"}),

    html.Div(id="data-preview-tab8"),

    html.Div([
        html.H3("Select Plot Type"),
        dcc.Dropdown(
            id="plot-type-dropdown",
            options=[
                {"label": "Bar Plot", "value": "bar"},
                {"label": "Count Plot", "value": "count"},
                {"label": "Pie Chart", "value": "pie"},
                {"label": "Box Plot", "value": "box"},
                {"label": "Violin Plot", "value": "violin"},
                {"label": "Strip Plot", "value": "strip"},
                {"label": "Swarm Plot", "value": "swarm"}
            ],
            placeholder="Select plot type",
        )
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.H3("Select Categorical Feature(s)"),
        dcc.Dropdown(
            id="feature-dropdown-tab8",
            multi=True,
            placeholder="Select categorical feature(s)"
        )
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.H3("Select Numerical Feature (if required)"),
        dcc.Dropdown(
            id="numerical-dropdown-tab8",
            multi=False,
            placeholder="Select a numerical feature"
        )
    ], style={"marginBottom": "20px"}),

    html.Button("Generate Plot", id="generate-plot-btn-tab8", n_clicks=0),

    html.Div(id="plot-container-tab8")
])

@app.callback(
    [Output("data-preview-tab8", "children"),
     Output("feature-dropdown-tab8", "options"),
     Output("numerical-dropdown-tab8", "options")],
    Input("upload-data-tab8", "contents"),
    State("upload-data-tab8", "filename"),
    prevent_initial_call=True
)
def load_data_tab8(contents, filename):
    global global_data
    if contents is None:
        return "No file uploaded", [], []

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith(".csv"):
            global_data = pd.read_csv(io.BytesIO(decoded))
        elif filename.endswith(".xlsx"):
            global_data = pd.read_excel(io.BytesIO(decoded))
        else:
            return "Unsupported file type", [], []

        categorical_columns = global_data.select_dtypes(include=["object", "category"]).columns
        numerical_columns = global_data.select_dtypes(include=["float64", "int64"]).columns

        categorical_options = [{"label": col, "value": col} for col in categorical_columns]
        numerical_options = [{"label": col, "value": col} for col in numerical_columns]

        preview = f"Loaded dataset with {global_data.shape[0]} rows and {global_data.shape[1]} columns."

        return preview, categorical_options, numerical_options
    except Exception as e:
        return f"Error loading file: {str(e)}", [], []

@app.callback(
    Output("plot-container-tab8", "children"),
    Input("generate-plot-btn-tab8", "n_clicks"),
    State("plot-type-dropdown", "value"),
    State("feature-dropdown-tab8", "value"),
    State("numerical-dropdown-tab8", "value"),
    prevent_initial_call=True
)
def generate_plot_tab8(n_clicks, plot_type, features, numerical_feature):
    global global_data

    if global_data is None or plot_type is None or not features:
        return "Please upload a dataset, select a plot type, and choose the required features."

    if plot_type == "bar":
        fig = px.bar(global_data, x=features[0], color=features[1] if len(features) > 1 else None)
    elif plot_type == "count":
        fig = px.histogram(global_data, x=features[0], color=features[1] if len(features) > 1 else None)
    elif plot_type == "pie":
        if len(features) != 1:
            return "Pie chart requires exactly 1 feature."
        fig = px.pie(global_data, names=features[0])
    elif plot_type == "box":
        if numerical_feature is None:
            return "Box plot requires a numerical feature."
        fig = px.box(global_data, x=features[0], y=numerical_feature, color=features[1] if len(features) > 1 else None)
    elif plot_type == "violin":
        if numerical_feature is None:
            return "Violin plot requires a numerical feature."
        fig = px.violin(global_data, x=features[0], y=numerical_feature, color=features[1] if len(features) > 1 else None, box=True)
    elif plot_type == "strip":
        if numerical_feature is None:
            return "Strip plot requires a numerical feature."
        fig = px.strip(global_data, x=features[0], y=numerical_feature, color=features[1] if len(features) > 1 else None)
    elif plot_type == "swarm":
        if numerical_feature is None:
            return "Swarm plot requires a numerical feature."
        fig = px.strip(global_data, x=features[0], y=numerical_feature, color=features[1] if len(features) > 1 else None, jitter=1)
    else:
        return "Plot type not implemented yet."

    return dcc.Graph(figure=fig)
# Tab 9 Layout
tab9_layout = html.Div([
    html.H1("Numerical Statistics and Correlation Heatmap"),
    html.Div([
        html.H3("Upload a CSV File"),
        dcc.Upload(
            id="upload-data-tab9",
            children=html.Button("Upload File"),
            multiple=False
        )
    ], style={"marginBottom": "20px"}),

    html.Div(id="data-preview-tab9"),

    html.Div([
        html.H3("Summary Statistics"),
        html.Div(id="stats-table-tab9"),
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.H3("Correlation Heatmap"),
        dcc.Graph(id="heatmap-tab9"),
    ])
])

@app.callback(
    [Output("data-preview-tab9", "children"),
     Output("stats-table-tab9", "children"),
     Output("heatmap-tab9", "figure")],
    Input("upload-data-tab9", "contents"),
    State("upload-data-tab9", "filename"),
    prevent_initial_call=True
)
def load_and_process_data_tab9(contents, filename):
    if contents is None:
        return "No file uploaded", None, {}

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:

        if filename.endswith(".csv"):
            tab9_data = pd.read_csv(io.BytesIO(decoded))
        elif filename.endswith(".xlsx"):
            tab9_data = pd.read_excel(io.BytesIO(decoded))
        else:
            return "Unsupported file type", None, {}

        numerical_data = tab9_data.select_dtypes(include=["float64", "int64"])

        if numerical_data.empty:
            return "No numerical data in the file.", None, {}

        preview = f"Loaded dataset with {numerical_data.shape[0]} rows and {numerical_data.shape[1]} numerical columns."

        stats = numerical_data.describe().reset_index()
        stats_table = html.Table(

            [html.Tr([html.Th(col) for col in stats.columns])] +

            [html.Tr([html.Td(stats.iloc[i][col]) for col in stats.columns]) for i in range(len(stats))]
        )


        corr = numerical_data.corr()
        heatmap_fig = px.imshow(
            corr,
            color_continuous_scale="viridis",
            title="Correlation Heatmap",
            labels=dict(x="Features", y="Features", color="Correlation"),
        )
        heatmap_fig.update_layout(xaxis=dict(tickmode="array"), yaxis=dict(tickmode="array"))

        return preview, stats_table, heatmap_fig

    except Exception as e:
        return f"Error loading file: {str(e)}", None, {}
# Main Layout
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='Data', children=[
        dcc.Tab(label='Data', value='Data'),
        dcc.Tab(label='Cleaning', value='Cleaning'),
        dcc.Tab(label='Outliers', value='Outliers'),
        dcc.Tab(label='PCA', value='PCA'),
        dcc.Tab(label='Normality', value='Normality'),
        dcc.Tab(label='Transformation', value='Transformation'),
        dcc.Tab(label='Numerical Features', value='Numerical Features'),
        dcc.Tab(label='Categorical Features', value='Categorical Features'),
        dcc.Tab(label='Numerical Statistics', value='Numerical Statistics'),
    ]),
    html.Div(id='tab-content')
])

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab_name):
    if tab_name == 'Data':
        return tab1_layout
    elif tab_name == 'Cleaning':
        return tab2_layout
    elif tab_name == 'Outliers':
        return tab3_layout
    elif tab_name == 'PCA':
        return tab4_layout
    elif tab_name == 'Normality':
        return tab5_layout
    elif tab_name == 'Transformation':
        return tab6_layout

    if tab_name == 'Numerical Features':
        return tab7_layout
    if tab_name == 'Categorical Features':
        return tab8_layout
    if tab_name == 'Numerical Statistics':
        return tab9_layout
    else:
        return html.Div("Tab not implemented.")
if __name__ == '__main__':
    app.run_server(debug=True)
