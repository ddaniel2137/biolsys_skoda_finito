import plotly.express as px
from numpy.random import RandomState
import numpy as np
import streamlit as st
from typing import List, Union
import pandas as pd
import inspect
import streamlit.components.v1 as stc
import re
import copy
from stqdm import stqdm
import os
from environment import Environment, run_simulation
from visualization import create_frames, plot_population_sizes, plot_fitnesses, build_figure, plot_population_contours
from utils import pad_sizes, preprocess_data
from parallel import search_optimal_parameters_parallel



def main():
    setup_interface()
    params = setup_sidebar_controls()
    params_required = list(inspect.signature(Environment.__init__).parameters.keys())[1:-1]
    params_required.extend(
        ['seed', 'meteor_impact_strategy', 'global_warming_scale', 'global_warming_var', 'meteor_impact_every',
         'meteor_impact_at'])
    params_provided = {k: st.session_state[k] for k in params_required}
    rng = RandomState(st.session_state['seed'])
    
    col = st.columns((1.5, 4.5, 2), gap='medium')
    col0_placeholder = col[0].empty()
    col2_placeholder = col[2].empty()
    
    with col0_placeholder:
        st.header('Prey parameters')
        st.markdown(f"Number of genes: {params['num_genes'][0]}")
        st.markdown(f"Fitness coefficients: {params['fitness_coefficients'][0]}")
        st.markdown(f"Mutation probabilities: {params['mutation_probabilities'][0]}")
        st.markdown(f"Mutation effects: {params['mutation_effects'][0]}")
        st.markdown(f"Max number of children: {params['max_num_children'][0]}")
        st.markdown(f"Interaction values: {params['interaction_values'][0]}")
        st.markdown(f"Initial populations: {params['init_populations'][0]}")
    
    with col[1]:
        
        st.header('Simulation')
        if st.button('🎲', key='dice_all'):
            st.session_state["random_all"] = True
            st.rerun()
        
        if "random_all" in st.session_state and st.session_state["random_all"]:
            st.session_state["random_all"] = False
        
        if st.button('Run Simulation'):
            st.session_state['run_simulation'] = True
        
        if 'run_simulation' in st.session_state and st.session_state['run_simulation']:
            try:
                env = Environment(**params_provided)
                _, stats_stacked = run_simulation(env, st.session_state['num_generations'])
                stats_df = preprocess_data(stats_stacked, params['roles'])
                # Assuming stats_stacked is structured as: { 'stat_name': [values_over_time], ... }
                # Direct conversion to DataFrame
                # df = preprocess_data(stats_stacked, params['roles'])
                # df_new = expand_variable_length_columns(df, 'element')
                # ic(df['genotypes'])
                # df.drop(columns=[('genotypes', 'prey'), ('genotypes', 'predator'), ('fitnesses', 'prey'), ('fitnesses', 'predator')], inplace=True)
                # ic(df['genotypes'])
                # ic(df)
                # ic(df.columns)
                # ic(df.head())
                # ic(df.genotypes)
                # ic(df.element_genotypes)
                # st.data_editor(df)
                display_results(stats_stacked)
                st.session_state['stats_df'] = stats_df
                st.session_state['stats_stacked'] = stats_stacked
                st.session_state['run_simulation'] = False
            
            except ValueError as ve:
                st.error(f"Failed to run simulation due to value error: {str(ve)}")
                st.session_state['run_simulation'] = False
            except KeyError as ke:
                st.error(f"Failed to run simulation due to key error: {str(ke)}")
                st.session_state['run_simulation'] = False
            except Exception as e:
                st.error(f"Failed to run simulation: {str(e)}")
                st.session_state['run_simulation'] = False
        
        if 'stats_df' in st.session_state:
            if st.button('Show animations'):
                st.session_state['show_animations'] = True
            
            if 'show_animations' in st.session_state and st.session_state['show_animations']:
                with st.form('animation_speed_form'):
                    animation_speed_prey = st.slider('Animation Speed for Prey (ms per frame)', min_value=0,
                                                     max_value=2000, value=800, step=100)
                    animation_speed_predator = st.slider('Animation Speed for Predator (ms per frame)', min_value=0,
                                                         max_value=2000, value=800, step=100)
                    submit_animation = st.form_submit_button('Submit')
                
                if submit_animation:
                    frames_prey = create_frames(st.session_state['stats_stacked'], 'prey')
                    frames_predator = create_frames(st.session_state['stats_stacked'], 'predator')
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.header('Prey Population')
                        fig_prey = build_figure(frames_prey, 'Prey', animation_speed_prey)
                        st.plotly_chart(fig_prey, use_container_width=True)
                    
                    with col2:
                        st.header('Predator Population')
                        fig_predator = build_figure(frames_predator, 'Predator', animation_speed_predator)
                        st.plotly_chart(fig_predator, use_container_width=True)
    
    with col2_placeholder:
        st.header('Predator parameters')
        st.markdown(f"Number of genes: {params['num_genes'][1]}")
        st.markdown(f"Fitness coefficients: {params['fitness_coefficients'][1]}")
        st.markdown(f"Mutation probabilities: {params['mutation_probabilities'][1]}")
        st.markdown(f"Mutation effects: {params['mutation_effects'][1]}")
        st.markdown(f"Max number of children: {params['max_num_children'][1]}")
        st.markdown(f"Interaction values: {params['interaction_values'][1]}")
        st.markdown(f"Initial populations: {params['init_populations'][1]}")
    
    st.header('Grid Search for Optimal Parameters')
    st.write(
        'This section performs a grid search to find the optimal parameter settings that maximize population survival across generations.')
    st.write(
        'The simulation is run with fixed parameters and tunable parameters are varied across a grid of values.')
    st.write('The results are displayed in a table for further analysis.')
    
    st.write('Do you wanna run the grid search?')
    if st.button('Yeah'):
        st.session_state['run_grid_search'] = True
    
    if 'run_grid_search' in st.session_state and st.session_state['run_grid_search']:
        st.session_state['run_grid_search'] = False
        if 'results_grid_mut' not in st.session_state:
            if os.path.exists('results_grid_mut.csv'):
                results_grid_mut = pd.read_csv('results_grid_mut.csv')
            else:
                
                results_grid_mut = search_optimal_parameters_parallel(
                    grid_params = {
                        'mutation_probabilities': (
                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        ),
                        'mutation_effects': (
                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        )
                    }
                )
                results_grid_mut.to_csv('results_grid_mut.csv', index=False)
            st.session_state['results_grid_mut'] = results_grid_mut
        else:
            pass
        
        if 'results_grid_coef' not in st.session_state:
            if os.path.exists('results_grid_coef.csv'):
                results_grid_coef = pd.read_csv('results_grid_coef.csv')
            else:
                results_grid_coef = search_optimal_parameters_parallel(
                grid_params = {
                    'fitness_coefficients': (
                        [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2.0, 5.0, 10.0, 20.0, 50.0, 75.0, 100.0],
                        [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2.0, 5.0, 10.0, 20.0, 50.0, 75.0, 100.0],
                        )
                    }
                )
                results_grid_coef.to_csv('results_grid_coef.csv', index=False)
            st.session_state['results_grid_coef'] = results_grid_coef
        else:
            pass

        if 'results_grid_interactions' not in st.session_state:
            if os.path.exists('results_grid_interactions.csv'):
                results_grid_interactions = pd.read_csv('results_grid_interactions.csv')
            else:
                results_grid_interactions = search_optimal_parameters_parallel(
                    grid_params = {
                        'interaction_values': (
                            [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        )
                    }
                )
                results_grid_interactions.to_csv('results_grid_interactions.csv', index=False)
            st.session_state['results_grid_interactions'] = results_grid_interactions
        else:
            pass

        st.dataframe(st.session_state['results_grid_mut'])
        st.dataframe(st.session_state['results_grid_coef'])
        st.dataframe(st.session_state['results_grid_interactions'])
        st.write('Grid search complete!')
        st.write('End of grid search results.')
        
    
    #show stats from grid runs
    if st.button('Show visual analysis'):
        st.session_state['show_visual_analysis'] = True
    
    # Example usage within the main application
    if 'show_visual_analysis' in st.session_state and st.session_state['show_visual_analysis']:
        st.session_state['show_visual_analysis'] = False

        col_contours = st.columns(3)
        with col_contours[0]:
            fig_prey, fig_predator = plot_population_contours(st.session_state['results_grid_coef'], 'fitness_coefficients', 'size', 'Fitness coefficients vs mean fitness', 'Fitness Coefficients (prey)', 'Fitness Coefficients (predator)')
            st.plotly_chart(fig_prey, use_container_width=True)
            st.plotly_chart(fig_predator, use_container_width=True)
        
        with col_contours[1]:
            fig_prey, fig_predator = plot_population_contours(st.session_state['results_grid_interactions'], 'interaction_values', 'size', 'Interaction Values vs mean fitness', 'Interaction Values (prey)', 'Interaction Values (predator)')
            st.plotly_chart(fig_prey, use_container_width=True)
            st.plotly_chart(fig_predator, use_container_width=True)

        with col_contours[2]:
            fig_prey_effect, fig_predator_effect = plot_population_contours(st.session_state['results_grid_mut'], 'mutation_effects', 'size', 'Mutation Effects vs mean fitness', 'Mutation effects (prey)', 'Mutation effects (predator)')
            st.plotly_chart(fig_prey_effect, use_container_width=True)
            st.plotly_chart(fig_predator_effect, use_container_width=True)
            fig_prey_chance, fig_predator_chance = plot_population_contours(st.session_state['results_grid_mut'], 'mutation_probabilities', 'size', 'Mutation Probabilities vs mean fitness', 'Mutation Probabilities (prey)', 'Mutation Probabilities (predator)')
            st.plotly_chart(fig_prey_chance, use_container_width=True)
            st.plotly_chart(fig_predator_chance, use_container_width=True)

def setup_interface():
    st.set_page_config(
        page_title="Population Simulation",
        layout="wide"
    )
    st.title('Evolutionary Simulation')
    st.write(
        'This is a simple simulation of evolution. The goal is to evolve a population of individuals to match a target genotype.')
    st.write(
        'The population evolves through mutation and reproduction, with genotypes evolving towards a target through simulated genetic processes.')
    # ... rest of your setup code


def display_results(stats_stacked):
    st.subheader('Population Sizes')
    sizes = [stats_stacked['size'][role] for role in ['prey', 'predator']]
    roles = {0: 'Prey', 1: 'Predator'}
    plot_population_sizes(sizes, roles)
    
    st.subheader('Fitnesses')
    fitnesses = [stats_stacked['mean_fitness'][role] for role in ['prey', 'predator']]
    roles = {0: 'Prey', 1: 'Predator'}
    plot_fitnesses(fitnesses, roles)


def create_slider_with_dice(label: str, min_value: Union[int, float, List[Union[int, float]]],
                            max_value: Union[int, float, List[Union[int, float]]],
                            default_value: Union[int, float, List[Union[int, float]]], key: str) -> Union[
    Union[int, float], List[Union[int, float]]]:
    """
    Creates sliders and dice buttons for randomization. Supports both single values and lists.

    Args:
        label (str): The label for the slider.
        min_value (Union[int, float, List[Union[int, float]]]): The minimum value for the slider.
        max_value (Union[int, float, List[Union[int, float]]]): The maximum value for the slider.
        default_value (Union[int, float, List[Union[int, float]]]): The default value(s) for the slider.
        key (str): The key to identify the slider.

    Returns:
        Union[Union[int, float], List[Union[int, float]]]: The value(s) corresponding to the slider(s).
    """
    # Normalize inputs using match case
    match min_value:
        case list() as lst:
            min_values = lst
        case _:
            min_values = [min_value] * (len(default_value) if isinstance(default_value, list) else 1)
    
    match max_value:
        case list() as lst:
            max_values = lst
        case _:
            max_values = [max_value] * (len(default_value) if isinstance(default_value, list) else 1)
    
    match default_value:
        case list() as lst:
            default_values = lst
        case _:
            default_values = [default_value]
    
    values = []
    for i, (min_val, max_val, def_val) in enumerate(zip(min_values, max_values, default_values)):
        slider_label = f"{label} {i + 1}" if len(default_values) > 1 else label
        slider_key = f"{key}_{i}"
        random_state = f"random_{slider_key}"
        if st.sidebar.button('🎲', key=f'dice_{slider_key}_button') or st.session_state.get('random_all',
                                                                                           False) and slider_key != 'num_populations_0':
            rng = np.random.default_rng()
            if isinstance(def_val, float):
                st.session_state[f"dice_{slider_key}"] = rng.uniform(min_val, max_val)
            else:
                st.session_state[f"dice_{slider_key}"] = rng.integers(int(min_val), int(max_val) + 1)
            st.session_state[random_state] = True
        
        if random_state in st.session_state and st.session_state[random_state]:
            def_val = copy.deepcopy(st.session_state[f"dice_{slider_key}"])
        
        value = st.sidebar.slider(slider_label, min_val, max_val, def_val, key=slider_key)
        
        if value != st.session_state[slider_key]:
            st.session_state[random_state] = False
        
        values.append(value)
    
    return values if len(values) > 1 else values[0]


def setup_sidebar_controls():
    params = {
        'roles': ['prey', 'predator'],
        'seed': None,
        'num_populations': None,
        'init_populations': None,
        'num_genes': None,
        'optimal_genotypes': None,
        'fitness_coefficients': None,
        'max_populations': None,
        'mutation_probabilities': None,
        'mutation_effects': None,
        'max_num_children': None,
        'interaction_values': None,
        'num_generations': None,
        'scenario': None,
        'meteor_impact_strategy': None,
        'global_warming_scale': None,
        'global_warming_var': None,
        'meteor_impact_every': None,
        'meteor_impact_at': None
    }
    roles = params['roles']
    
    params['seed'] = st.sidebar.number_input('Seed', 0, 1000, 42)
    params['num_populations'] = create_slider_with_dice('Number of populations', 1, 10, 2, 'num_populations')
    params['init_populations'] = create_slider_with_dice(f'Initial population', [1, 1], [1000, 1000], [200, 200],
                                                         'init_populations')
    params['num_genes'] = create_slider_with_dice(f'Number of genes', [2, 2], [10, 10], [5, 5], 'num_genes')
    params['fitness_coefficients'] = create_slider_with_dice(f'Fitness coefficient', [0.1, 0.1], [10.0, 10.0],
                                                             [0.75, 0.75], 'fitness_coefficients')
    params['max_populations'] = create_slider_with_dice(f'Max population', [100, 100], [10000, 10000], [1000, 1000],
                                                        'max_populations')
    params['mutation_probabilities'] = create_slider_with_dice(f'Mutation probability', [0.0, 0.0], [1.0, 1.0],
                                                               [0.15, 0.15], 'mutation_probabilities')
    params['mutation_effects'] = create_slider_with_dice(f'Mutation effect', [0.0, 0.0], [1.0, 1.0], [0.1, 0.1],
                                                         'mutation_effects')
    params['max_num_children'] = create_slider_with_dice(f'Max number of children', [1, 1], [10, 10], [2, 2],
                                                         'max_num_children')
    params['interaction_values'] = create_slider_with_dice(f'Interaction value', [-1.0, 0.0], [0.0, 1.0], [-0.6, 0.8],
                                                           'interaction_values')
    params['num_generations'] = create_slider_with_dice('Number of generations', 1, 1000, 300, 'num_generations')
    params['scenario'] = st.sidebar.selectbox('Scenario', ['global_warming', 'None'])
    params['meteor_impact_strategy'] = st.sidebar.selectbox('Meteor impact strategy', ['None', 'every', 'at'])
    if params['scenario'] == 'global_warming':
        params['global_warming_scale'] = create_slider_with_dice('Global warming scale', 0.0, 1.0, 1.0,
                                                                 'global_warming_scale')
        params['global_warming_var'] = create_slider_with_dice('Global warming variance', 0.0, 1.0, 0.05,
                                                               'global_warming_var')
    else:
        params['global_warming_scale'] = None
        params['global_warming_var'] = None
    
    if params['meteor_impact_strategy'] == 'every':
        params['meteor_impact_every'] = create_slider_with_dice('Meteor impact every', 1, 100, 20,
                                                                'meteor_impact_every')
        params['meteor_impact_at'] = None
    elif params['meteor_impact_strategy'] == 'at':
        params['meteor_impact_at'] = st.sidebar.multiselect('Meteor impact at',
                                                            list(range(1, params['num_generations'] + 1 if isinstance(params['num_generations'], int) else 100, 20)),
                                                            [20, 40])
        params['meteor_impact_every'] = None
    else:
        params['meteor_impact_every'] = None
        params['meteor_impact_at'] = None
    params['optimal_genotypes'] = [np.zeros(int(params['num_genes'][i]), dtype=float) if isinstance(params['num_genes'], list) else np.zeros(3, dtype=float) for i, _ in enumerate(roles)]
    for key, value in params.items():
        st.session_state[key] = value
    
    return params

# Refactored code
def get_fitness_range(default_range, custom_range_input):
    """Get the fitness range from the user input."""
    if custom_range_input:
        try:
            custom_range = set(map(float, re.split(r',\s*', custom_range_input)))
        except ValueError:
            st.error("Invalid input for fitness range. Please use comma separated float values.")
            return default_range
        custom_range.update(default_range)
        return custom_range
    return default_range


if __name__ == '__main__':
    main()
