import plotly.graph_objects as go
import numpy as np

fig = go.Figure(
    data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=[0, 0, 1, 1, 0, 0, 1, 1],
            y=[0, 1, 1, 0, 0, 1, 1, 0],
            z=[0, 0, 0, 0, 1, 1, 1, 1],
            # i, j and k give the vertices of triangles
            # i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            # j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            # k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            name='y',
            alphahull=0,
            showscale=True,
            opacity=0.5,
            color=0.6,
        )
    ]
)

fig.show()
