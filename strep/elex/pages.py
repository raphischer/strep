from dash import html, dcc
import dash_bootstrap_components as dbc

def create_axis_option(x=True):
    xy = 'x' if x else 'y'
    content = [
        html.Div(children=[dbc.Select(id=f'{xy}axis')]),
        html.Label('Weight:'),
        dcc.Input(id=f"{xy}-weight", type='number', min=0, max=1, step=0.05),
        html.Label('Boundaries:'),
        dcc.RangeSlider(id=f'boundary-slider-{xy}', min=0, max=1,
            value=[.2, .4, .6, .8], step=.001, pushable=.001,
            tooltip={"placement": "bottom", "always_visible": True})  
    ]

    return dbc.AccordionItem(content, title=f'{xy}-Axis Configuration')

style_btn_cfg = {
    'width': '90%',
    'textAlign': 'center'
}

style_upload = dict({
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
}, **style_btn_cfg)


def create_page(databases, indexmode, compound_mode, **kwargs):
    db = kwargs.get("database")
    if db is None or db not in databases.keys():
        db = list(databases.keys())[0]
    # if isinstance(meta_info, str) and os.path.isfile(meta_info):
    #     meta_info = read_json(meta_info)
    # if not isinstance(meta_info, dict):
    #     raise RuntimeError # TODO error msg
    
    # task configuration (offcanvas)
    task_configuration = dbc.Offcanvas(
        html.Div(children=[
            html.Div(children=[
                html.H2('Database:'),
                dbc.Select(id='db-switch', value=db, options=[{'label': db, 'value': db} for db in databases.keys()],)
            ]),
            html.Div(children=[
                html.H2('Dataset:'),
                dbc.Select(id='ds-switch')
            ]),
            html.Div(children=[
                html.H2('ML Task:'),
                dbc.RadioItems(id='task-switch')
            ]),
            html.Div(children=[
                html.H2('Environments:'),
                dbc.Checklist(id='environments')
            ]),
        ]),
        id="exp-config", title="Experiment Configuration", is_open=False,
    )

    # graph configuration (offcanvas)
    graph_configuration = dbc.Offcanvas(
        dbc.Accordion([
            create_axis_option(),
            create_axis_option(False),
            dbc.AccordionItem([
                html.H4('Scale to Use'),
                dbc.RadioItems(
                    id='scale-switch', value='index',
                    options=[
                        {'label': 'Index Scale', 'value': 'index'},
                        {'label': 'Value Scale', 'value': 'value'}
                    ],
                ),
                html.H4('Index Mode'),
                dbc.RadioItems(
                    id='indexmode-switch', value=indexmode,
                    options=[
                        {'label': 'Centered', 'value': 'centered', 'disabled': True},
                        {'label': 'Best', 'value': 'best'}
                    ],
                ),
                html.H4('Reference Model'),
                dbc.Select(id=f'select-reference'),
                html.H4('Boundaries and Weights'),
                dbc.Button("Select Optimal Reference", id="btn-optimize-reference", style=style_btn_cfg),
                dbc.Button("Calculate Optimal Boundaries", id="btn-calc-boundaries", active=False, style=style_btn_cfg),
                dbc.Button("Save Current Boundaries", id="btn-save-boundaries", style=style_btn_cfg),
                dcc.Download(id="save-boundaries"),
                dcc.Upload(
                    id="boundaries-upload", className='btn btn-default', style=style_btn_cfg,
                    children=['Drop or ', html.A('Select a Boundaries File (.json)')],
                ),
                dbc.Button("Save Current Metric Weights", id="btn-save-weights", style=style_btn_cfg),
                dcc.Download(id="save-weights"),
                dcc.Upload(
                    id="weights-upload", className='btn btn-default', style=style_btn_cfg,
                    children=['Drop or ', html.A('Select a Weights File (.json)')],
                ),
                html.H4('Compound Mode'),
                dbc.RadioItems(
                    id='compound_mode', value=compound_mode,
                    options=[{'label': opt, 'value': opt.lower()} for opt in ['Mean', 'Median', 'Min', 'Max']],
                )
            ], title = 'More Graph Options')
        ], start_collapsed=True), id="graph-config", title="Graph Configuration", is_open=False, style=dict(width='40%')
    )

    # graphs
    graph_scatter = dcc.Graph(
        id='graph-scatter',
        responsive=True,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'}
    )
    graph_bars = dcc.Graph(
        id='graph-bars',
        responsive=True,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'}
    )
    
    # label display & tables
    label_display = html.Div(children=[
        html.H3('Energy Label:'),
        html.Img(id='model-label', className="img-fluid"),
        dbc.Tooltip("Click to enlarge", target="model-label"),
    ])

    label_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Energy Label")),
            dbc.ModalBody(html.Img(id='label-modal-img', style={"height": "70vh"})),
            dbc.ModalFooter(dbc.Button("Save Label", id="btn-save-label2"))
        ],
        id="label-modal",
        is_open=False,
    )

    table_model = html.Div(children=[
        html.H3('General Information:'),
        dbc.Table(id='model-table', bordered=True)
    ])
    table_metrics = html.Div(children=[
        html.H3('Properties:'),
        dbc.Table(id='metric-table', bordered=True),
    ])

    buttons = dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Button("Open Paper", id="btn-open-paper", class_name='col-12'), width=3),
            dbc.Col(dbc.Button("Save Label", id="btn-save-label", class_name='col-12'), width=3),
            dbc.Col(dbc.Button("Save Summary", id="btn-save-summary", class_name='col-12'), width=3),
            dbc.Col(dbc.Button("Save Logs", id="btn-save-logs", class_name='col-12'), width=3)
        ], style={'height': '50px'}, className="g-1"),
        # config buttons
        dbc.Row([
            dbc.Col(dbc.Button("Task Configuration", id="btn-open-exp-config", size="lg", class_name='col-12'), width=6),
            dbc.Col(dbc.Button("Graph Configuration", id="btn-open-graph-config", size="lg", class_name='col-12'), width=6)
        ], style=dict(height='50px'), className="g-1")
    ], style={'padding': '0 0'})

    info_hover = dbc.Alert(
        "Hover over data points to show model information",
        id="info-hover",
        dismissable=True,
        is_open=True,
        color="info",
    )
    
    row0 = [
        dbc.Col(html.H1('Sustainable and Trustworthy ML Reporting')),
        dbc.Col(html.Img(src="assets/lamarr-logo-2023-negative.svg", className="img-fluid"), width=2)
    ]

    row1 = [
        dbc.Col(graph_scatter, width=8),
        dbc.Col(graph_bars, width=4)
    ]

    row2 = [
        dbc.Col([table_model, buttons], width=5),
        dbc.Col(table_metrics, width=5),
        dbc.Col(label_display, width=2)
    ]
    
    return html.Div([
        dcc.Location(id="url", refresh=False),
        dbc.Container([
            dbc.Row(row0, style={"height": "15vh"}, align="center"),
            dbc.Row(row1, style={"height": "40vh"}),
            dbc.Row(info_hover),
            dbc.Row(row2, style={"height": "50vh"}),
        ]),
        # additional hidden html elements
        label_modal, task_configuration, graph_configuration, dcc.Download(id="save-label"),
    ])
