import dash_bootstrap_components as dbc


class Panel:
    def __init__(self, width, name):
        self.width = width
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.children.append(child)

    def empty_panel(self):
        self.children = []

    def get_html(self):
        total_height = 0
        max_height = 99
        for child in self.children:
            total_height += child.default_height
        ratio = max_height/total_height
        for child in self.children:
            child.height = child.default_height * ratio

        return dbc.Col(id=self.name,
                       children=[child.get_html() for child in self.children],
                       width={"size": self.width})

    def create_children_html(self):
        return [child.get_html() for child in self.children]
