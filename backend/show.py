import requests
from PIL import Image
import plotly.express as px
import altair as alt


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


def plot_caption_clusters(df, x, y, hover_data, color, size, text, opacity=0.7):
    fig = px.scatter(df, x=x, y=y, hover_data=hover_data, color=color, opacity=opacity, size=size, text=text)

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),  # set the margins of the plot
        height=600,
        width=1050,
        title='Clustering Captions',  # set the title of the plot

        font=dict(family='Arial', size=12),  # set the font family and size
        showlegend=True,  # show the legend
        legend_title='Caption Category',  # set the title of the legend
        # template='plotly_white',
        legend_font=dict(family='Arial', size=10),  # set the font family and size of the legend

    )
    fig.update_traces(hovertemplate="<b>Category:</b> %{customdata[0]}<br>" +
                                    "<b>Caption:</b> %{customdata[1]}<br>")

    return fig

def plot_image_clusters(df):
    marker_chart = alt.Chart(df.rename(columns={'url': 'image'})).mark_circle(size=40).encode(
                    x='x',
                    y='y',
                    # size='size',
                    # color='artist',
                    tooltip=['image', 'artist']).properties(
                    width=800,
                    height=600, title='Clustering Images from https://www.rockarchive.com/',
                )
    return marker_chart







