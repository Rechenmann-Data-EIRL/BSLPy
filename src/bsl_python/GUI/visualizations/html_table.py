import dash_html_components as html
import dash_table


class HTMLTable:
    name = ""
    data = None

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def get_html(self):
        return html.Div(
            id=self.name,
            children=[dash_table.DataTable(
                id=self.name + "_table",
                columns=[{"name": i, "id": i} for i in self.data.columns],
                data=self.data.to_dict('records'),
                style_cell={"color": "black",
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'maxWidth': 0}
            )])
