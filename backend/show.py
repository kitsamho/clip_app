import requests
from PIL import Image
import plotly.express as px


def open_image(url):
    # Set the URL of a random image to fetch
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def plot_results(df, y_label, x_label):
    fig = px.bar(df, y=y_label, x=x_label, range_y=[0, 1], height=300, width=400)

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)', })

    return fig

