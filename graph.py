import plotly as py
import plotly.graph_objs as go
import numpy as np

class Graph:
    def __init__(self, tSteps, y1s, y2s, learnType, epsilon, gamma, alpha, nGames):
        self.line1 = np.array(y1s)
        self.line2 = np.array(y2s)
        self.xarray = np.array(tSteps)
        self.learnType = learnType
        self.layout = go.Layout(
            title='Random learning curve vs. RL curve: ' + learnType + '\n Epsilon: ' + str(epsilon) + ', Gamma: ' + str(gamma) + ', Alpha: ' + str(alpha) + '\n Number of Games: ' + str(nGames),
            xaxis=dict(
                title='Number of games played',
                titlefont=dict(
                    size=18,
                    color='#7f7f7f'
                )
            ),
            yaxis=dict(
                title='Average score per game',
                titlefont=dict(
                    size=18,
                    color='#7f7f7f'
                )
            )
        )

    def plot(self):
        py.tools.set_credentials_file(username='nicolepilsworth', api_key='RIr45yea9BGg46iuh29e')


        # Code used from https://plot.ly/python/line-charts/#new-to-plotly
        trace1 = go.Scatter(
            name="Random",
            x = self.xarray,
            y = self.line1
        )

        trace2 = go.Scatter(
            name=self.learnType,
            x = self.xarray,
            y = self.line2
        )

        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=self.layout)

        py.plotly.plot(fig, filename='basic-line')
        print("Lines plotted")
