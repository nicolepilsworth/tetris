import plotly as py
import plotly.graph_objs as go
import numpy as np

class PgGraph:
    def __init__(self, tSteps, ys, batch_size, maxPerEpisode, nGames, boardSize):
        self.line1 = np.array(ys)
        self.xarray = np.array(tSteps)
        self.layout = go.Layout(
            title='Policy Gradient curve\n Batch size: ' + str(batch_size) + ', Max moves per game: ' + str(maxPerEpisode) + '\n Number of Games: ' + str(nGames) + ", Board Size: " + boardSize,
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
            x = self.xarray,
            y = self.line1
        )

        data = [trace1]
        fig = go.Figure(data=data, layout=self.layout)

        py.plotly.plot(fig, filename='basic-line')
        print("Lines plotted")

    def plot(self):
        py.tools.set_credentials_file(username='nicolepilsworth', api_key='RIr45yea9BGg46iuh29e')

        # Code used from https://plot.ly/python/line-charts/#new-to-plotly
        trace1 = go.Scatter(
            x = self.xarray,
            y = self.line1
        )

        data = [trace1]
        fig = go.Figure(data=data, layout=self.layout)

        py.plotly.plot(fig, filename='basic-line')
        print("Lines plotted")
