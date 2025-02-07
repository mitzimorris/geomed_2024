import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, W
from typing import (Dict, List)

def disconnect_nbs(
        nbs: W, target_indices: List[int], remove_indices: List[int]
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

def nyc_cleanup(nbs: W, gdf: gpd.GeoDataFrame) -> W:
    """
    Modify neighbor graph of NYC to remove neighbor pairs between
    Manhattan and other boroughs (Brooklyn, Queens).
    - param nbs: neighbor graph
    - param gdf : geopandas.GeoDataFrame of NYC Census Tract data    
    """
    # Get indices for each borough
    manhattan_indices = gdf[gdf['BoroName'] == 'Manhattan'].index
    brooklyn_indices = gdf[gdf['BoroName'] == 'Brooklyn'].index
    queens_indices = gdf[gdf['BoroName'] == 'Queens'].index
    brooklyn_and_queens = list(brooklyn_indices) + list(queens_indices)
    # Disconnect Manhattan from Brooklyn/Queens and vice versa
    disconnect_nbs(nbs, manhattan_indices, brooklyn_and_queens)
    disconnect_nbs(nbs, brooklyn_indices, manhattan_indices)
    disconnect_nbs(nbs, queens_indices, manhattan_indices)

    return W(nbs.neighbors, nbs.weights)


def nyc_sort_by_comp_size(nyc_gdf: gpd.GeoDataFrame) -> tuple[W, gpd.GeoDataFrame, List[int]]:
    """
    Process NYC geodataframe - sort by component size descending
    - param nyc_gdf : geopandas.GeoDataFrame
    - return: tuple containing:
    """
    # Compute initial neighborhood graph
    nyc_nbs = Queen.from_dataframe(nyc_gdf, geom_col='geometry')
    # Clean borough connections and get components
    nyc_nbs_tmp  = nyc_cleanup(nyc_nbs, nyc_gdf)
    # Add component info to dataframe
    nyc_gdf['comp_id'] = nyc_nbs_tmp.component_labels
    sizes = nyc_gdf['comp_id'].value_counts()
    nyc_gdf['comp_size'] = nyc_gdf['comp_id'].map(sizes)
    # Sort by component size and reset index
    nyc_gdf_sorted = (nyc_gdf.sort_values(by='comp_size', ascending=False)
                             .reset_index(drop=True)
                             .set_geometry('geometry'))
    # Recompute neighborhood graph, update gdf, get component sizes
    nyc_nbs_sorted = Queen.from_dataframe(nyc_gdf_sorted, geom_col='geometry')
    nyc_nbs_clean = nyc_cleanup(nyc_nbs_sorted, nyc_gdf_sorted)
    nyc_gdf_sorted['comp_id'] = nyc_nbs_clean.component_labels
    component_sizes = list(sizes.sort_values(ascending=False))

    return nyc_nbs_clean, nyc_gdf_sorted, component_sizes


def connect_nbs(nbs: W, region_i: int, region_j: int) -> None:
    """
    Modify a neighbors graph to add a bidirectional connection between two nodes if one doesn't already exist.
    - param nbs : neighbor graph to modify
    - params region_i, region_j : IDs of nodes to connect
    """
    weight = 1.0
    # using R style node ids; Python counts from 0
    node_i = region_i - 1
    node_j = region_j - 1
    # Add bidirectional connection if it doesn't exist
    if node_j not in nbs.neighbors[node_i]:
        # Connect i -> j
        nbs.neighbors[node_i].append(node_j)
        nbs.weights[node_i].append(weight)
        # Connect j -> i 
        nbs.neighbors[node_j].append(node_i)
        nbs.weights[node_j].append(weight)

def connect_nyc(nyc_gdf: gpd.GeoDataFrame) -> W:
    nyc_nbs = Queen.from_dataframe(nyc_gdf, geom_col='geometry')
    connect_nbs(nyc_nbs, 1995, 387)  # Staten Island to Bay Ridge
    connect_nbs(nyc_nbs, 1861, 1863) # Breezy Point to Rockaways
    connect_nbs(nyc_nbs, 1904, 1859) # Broad Channel to Brooklyn
    connect_nbs(nyc_nbs, 1904, 1871) # Broad Channel to Rockaways
    connect_nbs(nyc_nbs, 1311, 1364) # Roosevelt Island to Queens
    connect_nbs(nyc_nbs, 1343, 193) # Manhattan to Bronx
    connect_nbs(nyc_nbs, 329, 212) # City Island to Bronx
    return W(nyc_nbs.neighbors, nyc_nbs.weights)
