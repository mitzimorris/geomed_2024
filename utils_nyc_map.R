library(sf)
library(dplyr)
library(spdep)

#' Disconnect neighbors in a neighbor list
#' 
#' @param nb Neighbor list object
#' @param target_indices Vector of node IDs to process
#' @param remove_indices Vector of non-neighbor node IDs to remove
#' @return Modified neighbor list
disconnect_nbs <- function(nb, target_indices, remove_indices) {
  remove_set <- remove_indices
  
  for (node in target_indices) {
    # Get current neighbors
    curr_neighbors <- nb[[node]]
    
    # Filter out removed neighbors
    clean_neighbors <- curr_neighbors[!curr_neighbors %in% remove_set]
    
    # Update neighbors
    nb[[node]] <- clean_neighbors
  }
  
  return(nb)
}

#' Clean NYC neighbor connections
#' 
#' @param nb Neighbor list object
#' @param sf_data SF dataframe with BoroName column
#' @return Modified neighbor list
nyc_cleanup <- function(nb, sf_data) {
  # Get indices for each borough
  manhattan_indices <- which(sf_data$BoroName == "Manhattan")
  brooklyn_indices <- which(sf_data$BoroName == "Brooklyn")
  queens_indices <- which(sf_data$BoroName == "Queens")
  
  # Combine Brooklyn and Queens indices
  brooklyn_and_queens <- c(brooklyn_indices, queens_indices)
  
  # Disconnect Manhattan from Brooklyn/Queens and vice versa
  nb <- disconnect_nbs(nb, manhattan_indices, brooklyn_and_queens)
  nb <- disconnect_nbs(nb, brooklyn_indices, manhattan_indices)
  nb <- disconnect_nbs(nb, queens_indices, manhattan_indices)
  
  return(nb)
}

#' Connect two regions in neighbor list
#' 
#' @param nb Neighbor list object
#' @param region_i First node ID
#' @param region_j Second node ID
#' @return Modified neighbor list
connect_nbs <- function(nb, region_i, region_j) {
  # Add bidirectional connection if it doesn't exist
  if (!region_j %in% nb[[region_i]]) {
    # Connect i -> j
    nb[[region_i]] <- c(nb[[region_i]], region_j)
    
    # Connect j -> i
    nb[[region_j]] <- c(nb[[region_j]], region_i)
  }
  
  return(nb)
}

#' Process NYC census tract data
#' 
#' @param nyc_sf SF dataframe with geometry and BoroName columns
#' @return List containing:
#'   - nb: Cleaned neighbor list
#'   - sf_sorted: Sorted census tract data
#'   - component_sizes: Component sizes in descending order
nyc_sort_by_comp_size <- function(nyc_sf) {
  # Compute initial neighborhood graph
  nyc_nb <- poly2nb(nyc_sf)
  
  # Clean borough connections
  nyc_nb <- nyc_cleanup(nyc_nb, nyc_sf)
  
  # Get components
  comp_info <- n.comp.nb(nyc_nb)
  nyc_sf$comp_id <- comp_info$comp.id
  
  # Calculate component sizes
  comp_sizes <- table(nyc_sf$comp_id)
  nyc_sf$comp_size <- comp_sizes[match(nyc_sf$comp_id, names(comp_sizes))]
  
  # Sort by component size
  nyc_sf_sorted <- nyc_sf %>%
    arrange(desc(comp_size)) %>%
    st_as_sf()
  
  # Recompute neighborhood graph for sorted data
  nyc_nb_sorted <- poly2nb(nyc_sf_sorted)
  nyc_nb_clean <- nyc_cleanup(nyc_nb_sorted, nyc_sf_sorted)
  
  # Update component labels
  comp_info_clean <- n.comp.nb(nyc_nb_clean)
  nyc_sf_sorted$comp_id <- comp_info_clean$comp.id
  
  # Get sorted component sizes
  component_sizes <- sort(table(nyc_sf_sorted$comp_id), decreasing = TRUE)
  
  return(list(
    nb = nyc_nb_clean,
    sf_sorted = nyc_sf_sorted,
    component_sizes = as.vector(component_sizes)
  ))
} 