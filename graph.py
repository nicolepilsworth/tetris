import plotly as py
import plotly.graph_objs as go
import numpy as np

class Graph:
    def __init__(self, tSteps, y1s, y2s):
        self.line1 = np.array(y1s)
        self.line2 = np.array(y2s)
        self.xarray = np.array(tSteps)

    def plot(self):
        py.tools.set_credentials_file(username='nicolepilsworth', api_key='RIr45yea9BGg46iuh29e')

        # Code used from https://plot.ly/python/line-charts/#new-to-plotly
        trace1 = go.Scatter(
            name="Random",
            x = self.xarray,
            y = self.line1
        )

        trace2 = go.Scatter(
            name="Q-learning",
            x = self.xarray,
            y = self.line2
        )

        data = [trace1, trace2]

        py.plotly.plot(data, filename='basic-line')
        print("Lines plotted")
