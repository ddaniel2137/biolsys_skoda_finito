from typing import List, Dict, Any, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st
from plotly import graph_objects as go
from plotly.graph_objs import Scatter, Frame, Layout, Figure
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from icecream import ic
from utils import pad_sizes


@st.cache_data
def create_frames(stats_stacked: Dict[str, Dict[str, List[Any]]], role: str) -> List[Frame]:
    frames = []
    all_genotypes = np.vstack([g for g in stats_stacked['genotypes'][role] if g.size > 0])
    all_optimal_genotypes = np.vstack(stats_stacked['optimal_genotype'][role])
    lengths = np.cumsum([0, *[g.shape[0] for g in stats_stacked['genotypes'][role]]])
    pca = TruncatedSVD(n_components=2)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    all_genotypes_scaled = scaler.fit_transform(pca.fit_transform(np.vstack([all_genotypes, all_optimal_genotypes])))
    all_genotypes_scaled, all_optimal_genotypes_scaled = np.split(all_genotypes_scaled, [len(all_genotypes)])
    #ic(all_genotypes_scaled)
    #ic(all_optimal_genotypes_scaled)
    for gen in range(len(stats_stacked['generation'][role])):
        pca_population = all_genotypes_scaled[lengths[gen]:lengths[gen + 1]]
        #ic(pca_population, pca_population.shape)
        pca_optimal = all_optimal_genotypes_scaled[gen, :].reshape(1, -1)
        #ic(pca_optimal, pca_optimal.shape)
        if pca_population.size > 0:
            frame_data = [
                Scatter(x=pca_population[:, 0], y=pca_population[:, 1], mode='markers', name='Population',
                        marker={'color': stats_stacked['fitnesses'][role][gen], 'size': 8, 'colorscale': 'Inferno',
                                'cmin': 0.0, 'cmid': 0.5, 'cmax': 1.0, 'opacity': 0.5,  # 'Viridis
                                'colorbar': {'title': 'Fitness'}}),
                Scatter(x=[pca_optimal[0, 0]], y=[pca_optimal[0, 1]], mode='markers', name='Optimal',
                        marker={'color': 'green', 'size': 12, 'symbol': 'cross'})
            ]
        else:
            frame_data = [
                Scatter(x=[pca_optimal[0, 0]], y=[pca_optimal[0, 1]], mode='markers', name='Optimal (only)',
                        marker={'color': 'red', 'size': 12, 'symbol': 'star'})
            ]
        
        frame = Frame(
            data=frame_data,
            layout=Layout(title=f"{role.capitalize()} population genotypes evolution", annotations=[{
                'text': f"Generation {gen} (optimal only)", 'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.95,
                'xanchor': 'center', 'yanchor': 'bottom', 'font': {'size': 16}
            }] if pca_population.size == 0 else [{
                'text': f"Generation {gen}", 'showarrow': False, 'xref': 'paper', 'yref': 'paper', 'x': 0.5,
                'y': 0.95,
                'xanchor': 'center', 'yanchor': 'bottom', 'font': {'size': 16}
            }])
        )
        frames.append(frame)
    
    return frames


def plot_population_sizes(sizes: List[np.ndarray], roles: Dict[int, str]) -> None:
    max_size = max([len(size) for size in sizes])
    padded_sizes = pad_sizes(sizes, max_size)
    fig = go.Figure()
    for i, size in enumerate(padded_sizes):
        fig.add_trace(go.Scatter(x=list(range(max_size)), y=size, mode='lines', name=roles[i]))
    fig.update_layout(title='Population Sizes', xaxis_title='Generation', yaxis_title='Size')
    st.plotly_chart(fig)


def plot_fitnesses(fitnesses: List[np.ndarray], roles: Dict[int, str]) -> None:
    max_size = max([len(fitness) for fitness in fitnesses])
    padded_fitnesses = pad_sizes(fitnesses, max_size)
    fig = go.Figure()
    for i, fitness in enumerate(padded_fitnesses):
        fig.add_trace(go.Scattergl(x=list(range(max_size)), y=fitness, mode='lines', name=roles[i]))
    fig.update_layout(title='Fitnesses', xaxis_title='Generation', yaxis_title='Fitness')
    st.plotly_chart(fig)


@st.cache_data
def build_figure(frames: List[Frame], role: str, animation_speed: int=400) -> go.Figure:
    if frames:
        initial_frame = frames[0]
        fig = Figure(
            data=initial_frame.data,
            layout=Layout(
                autosize=True,
                xaxis=dict(range=[-1, 1], autorange=False, zeroline=True, showgrid=False),
                yaxis=dict(range=[-1, 1], autorange=False, zeroline=True, showgrid=False),
                title=f"{role.capitalize()} Population Genotypes Evolution",
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {'frame': {'duration': animation_speed, 'redraw': True},
                                            'fromcurrent': True, 'transition': {'duration': 0}}]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                              'mode': 'immediate', 'transition': {'duration': 0}}]
                        },
                        {
                            'label': 'Reset',
                            'method': 'animate',
                            'args': [{'frame': {'duration': 0, 'redraw': True}},
                                     {'mode': 'immediate', 'transition': {'duration': 0}}],
                        }
                    ],
                    'direction': 'right',
                    'x': 1,
                    'xanchor': 'right',
                    'y': -0.3,
                    'yanchor': 'bottom'
                }]
            ),
            frames=frames
        )
        return fig
    else:
        return Figure()

def plot_population_contours(df: pd.DataFrame, coord_axis: str, z_axis: str, title: str, x_label: str, y_label: str) -> Tuple[Figure, Figure]:
    """
    Plots a contour chart for population data across specified indices and columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the simulation results.
    coord_axis (str): Column containing lists of [x, y] coordinates for each row.
    z_axis (str): Column to use as the z-axis (values).
    title (str): Title for the plot.
    x_label (str): Label for the X-axis.
    y_label (str): Label for the Y-axis.
    """
    
    df_temp = pd.DataFrame()
    df_temp[['x', 'y']] = df[coord_axis].apply(pd.Series)
    df_temp['role'] = df['role']
    df_temp[z_axis] = df[z_axis]
    
    fig_prey = go.Figure()
    fig_predator = go.Figure()

    for role in ['prey', 'predator']:
        df_role = df_temp[df_temp['role'] == role]
        
        fig = go.Figure(data=go.Contour(
            z=df_role[z_axis],
            x=df_role['x'],
            y=df_role['y'],
            colorscale='Viridis',
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=12, color='white')
            ),
            colorbar=dict(title=z_axis)
        ))
        
        fig.update_layout(
            title=f"{title} of {role.capitalize()}s",
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=800,
            height=600
        )
        
        if role == 'prey':
            fig_prey = fig
        else:
            fig_predator = fig
    
    return fig_prey, fig_predator


    

