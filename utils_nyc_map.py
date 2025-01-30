import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Rook, Queen, W
from typing import (Dict, List)

def disconnect_nbs(
        nbs: Rook, target_indices: List[int], remove_indices: List[int]
        ) -> None:
    """
    Modify a neighbors graph by removing indices from the list of neighbors
    and resetting neighbors, weights lists accordingly.
    - param nbs: neighbor graph
    - param target_indices: list of node ids to process 
    - param remove_indices: list of non-neighbor node ids
    """
    remove_set = set(remove_indices)  # Convert to set for O(1) lookups
    for node in target_indices:
        # Filter neighbors and weights in a single pass
        clean_neighbors = []
        clean_weights = []
        for neighbor, weight in zip(nbs.neighbors[node], nbs.weights[node]):
            if neighbor not in remove_set:
                clean_neighbors.append(neighbor)
                clean_weights.append(weight)
        nbs.neighbors[node] = clean_neighbors
        nbs.weights[node] = clean_weights

def nyc_cleanup(nbs: Rook, gdf: gpd.GeoDataFrame) -> None:
    """
    Modify neighbor graph of NYC to remove neighbor pairs between
    Manhattan and other boroughs (Brooklyn, Queens).
    
    Parameters
    ----------
    nbs : libpysal.weights.Rook
        Neighborhood graph based on gdf row indices
    gdf : geopandas.GeoDataFrame 
        Census tract labels and geometry with 'BoroName' column
    """
    # Get indices for each borough
    manhattan_indices = gdf[gdf['BoroName'] == 'Manhattan'].index
    brooklyn_indices = gdf[gdf['BoroName'] == 'Brooklyn'].index
    queens_indices = gdf[gdf['BoroName'] == 'Queens'].index
    
    # Simple list concatenation alternative
    brooklyn_and_queens = list(brooklyn_indices) + list(queens_indices)
    
    # Disconnect Manhattan from Brooklyn/Queens and vice versa
    disconnect_nbs(nbs, manhattan_indices, brooklyn_and_queens)
    disconnect_nbs(nbs, brooklyn_indices, manhattan_indices)
    disconnect_nbs(nbs, queens_indices, manhattan_indices)

def nyc_sort_by_comp_size(nyc_gdf: gpd.GeoDataFrame, metric: str) -> tuple[W, gpd.GeoDataFrame, List[int]]:
    """
    Process NYC census tract data by:
    1. Computing neighborhood graph from geometries
    2. Cleaning connections between boroughs
    3. Sorting tracts by connected component size
    
    Parameters
    ----------
    nyc_gdf : geopandas.GeoDataFrame
        Census tract data with 'geometry' and 'BoroName' columns
        
    Returns
    -------
    tuple containing:
        - W: Cleaned neighborhood graph
        - GeoDataFrame: Sorted census tract data
        - List[int]: Component sizes in descending order
    """
    # Compute initial neighborhood graph
    if metric == "queen":
        nyc_nbs = Queen.from_dataframe(nyc_gdf, geom_col='geometry')
    else:
        nyc_nbs = Rook.from_dataframe(nyc_gdf, geom_col='geometry')
    
    # Clean borough connections and get components
    nyc_cleanup(nyc_nbs, nyc_gdf)
    nyc_nbs_tmp = W(nyc_nbs.neighbors, nyc_nbs.weights)
    
    # Add component info to dataframe
    nyc_gdf['comp_id'] = nyc_nbs_tmp.component_labels
    sizes = nyc_gdf['comp_id'].value_counts()
    nyc_gdf['comp_size'] = nyc_gdf['comp_id'].map(sizes)
    
    # Sort by component size and reset index
    nyc_gdf_sorted = (nyc_gdf.sort_values(by='comp_size', ascending=False)
                             .reset_index(drop=True)
                             .set_geometry('geometry'))
    
    # Recompute neighborhood graph for sorted data
    if metric == "queen":
        nyc_nbs_sorted = Queen.from_dataframe(nyc_gdf_sorted, geom_col='geometry')
    else:
        nyc_nbs_sorted = Rook.from_dataframe(nyc_gdf_sorted, geom_col='geometry')
    nyc_cleanup(nyc_nbs_sorted, nyc_gdf_sorted)
    nyc_nbs_clean = W(nyc_nbs_sorted.neighbors, nyc_nbs_sorted.weights)
    
    # Update component labels and return
    nyc_gdf_sorted['comp_id'] = nyc_nbs_clean.component_labels
    component_sizes = list(sizes.sort_values(ascending=False))
    
    return nyc_nbs_clean, nyc_gdf_sorted, component_sizes

def connect_nbs(nbs: Rook, region_i: int, region_j: int) -> None:
    """
    Modify a neighbors graph to add a bidirectional connection between two nodes if one doesn't already exist.
    
    Parameters
    ----------
    nbs : libpysal.weights.Rook
        The neighbor graph to modify
    region_i : int
        First node ID to connect
    region_j : int 
        Second node ID to connect
    """
    weight = 1.0
    
    # Add bidirectional connection if it doesn't exist
    if region_j not in nbs.neighbors[region_i]:
        # Connect i -> j
        nbs.neighbors[region_i].append(region_j)
        nbs.weights[region_i].append(weight)
        nbs.cardinalities[region_i] = len(nbs.neighbors[region_i])
        
        # Connect j -> i 
        nbs.neighbors[region_j].append(region_i)
        nbs.weights[region_j].append(weight)
        nbs.cardinalities[region_j] = len(nbs.neighbors[region_j])

