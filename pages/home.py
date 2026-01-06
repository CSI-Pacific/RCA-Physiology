
import dash
from dash import html, dcc, callback, Input, Output

# Register this file as a page in the larger app
dash.register_page(__name__, path="/home", name="Home")

layout = html.Div([
    html.H1('Invervals Programming Home')
])