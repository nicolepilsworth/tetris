import plotly.plotly as py
import plotly
from plotly.graph_objs import *
import numpy as np

class Graph:
    def __init__(self, t_steps, data, x_title, y_title):
        self.data = data
        self.layout = Layout(
            xaxis=dict(
                title=x_title,
                range=t_steps,
                titlefont=dict(
                    size=18,
                    color='#7f7f7f'
                )
            ),
            yaxis=dict(
                title=y_title,
                titlefont=dict(
                    size=18,
                    color='#7f7f7f'
                )
            )
        )

    def plot(self):
        plotly.tools.set_credentials_file(username='nicolepilsworth', api_key='RIr45yea9BGg46iuh29e')

        plot_data = list(map(lambda x: Scatter(**x), self.data))
        fig = Figure(data=plot_data, layout=self.layout)

        py.plot(fig, filename='epsilon-tabq-graph')
        print("Lines plotted")