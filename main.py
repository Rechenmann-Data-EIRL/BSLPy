from src.bsl_python.GUI.app import app
from src.bsl_python.GUI.index import Index

from src.bsl_python.launcher.launcher import Launcher

app.layout = Index().get_html()

if __name__ == '__main__':
    window = Launcher()
    window.mainloop()
