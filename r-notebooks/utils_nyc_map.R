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

  # Remove zero entries from nbs
  singletons <- length(component_sizes[component_sizes == 1])
  nyc_nbs <- nyc_nb_clean[1:(length(nyc_nb_clean)-singletons)]
  attr(nyc_nbs, "region.id") <- attr(nyc_nbs, "region.id")[1:(length(nyc_nbs))]
  attr(nyc_nbs, "type") <- "queen"
  attr(nyc_nbs, "sym") <- TRUE  
  class(nyc_nbs) <- "nb"  

  return(list(
    nb = nyc_nbs,
    sf_sorted = nyc_sf_sorted,
    component_sizes = as.vector(component_sizes)
  ))
}

add_symmetric_edge_nb <- function(nb, i, j) {
  nb_res = nb
  if (!(j %in% nb[[i]])) {
    if (nb[[i]][1] == 0) {
      nb_res[[i]] = j
    } else {
      nb_res[[i]] <- sort(c(nb[[i]], j))
    }
  }
  if (!(i %in% nb[[j]])) {
    if (nb[[j]][1] == 0) {
      nb_res[[j]] = i
     } else {
       nb_res[[j]] <- sort(c(nb[[j]], i))
     }
  }
  return(nb_res)
}

# create fully connected network
# assumes nyc_gdf is from file "nyc_study.geojson", order unchanged
connect_nyc <- function(nyc_gdf) {
    nyc_cnbs = poly2nb(nyc_geodata)
    nyc_cnbs = add_symmetric_edge_nb(nyc_cnbs, 1995L, 387L)  # Staten Island to Bay Ridge
    nyc_cnbs = add_symmetric_edge_nb(nyc_cnbs, 1861L, 1863L) # Breezy Point to Rockaways
    nyc_cnbs = add_symmetric_edge_nb(nyc_cnbs, 1904L, 1859L) # Broad Channel to Brooklyn
    nyc_cnbs = add_symmetric_edge_nb(nyc_cnbs, 1904L, 1871L) # Broad Channel to Rockaways
    nyc_cnbs = add_symmetric_edge_nb(nyc_cnbs, 1311L, 1364L) # Roosevelt Island to Queens
    nyc_cnbs = add_symmetric_edge_nb(nyc_cnbs, 1343L, 193L) # Manhattan to Bronx
    nyc_cnbs = add_symmetric_edge_nb(nyc_cnbs, 329L, 212L) # City Island to Bronx
    return(nyc_cnbs)
}

# used to create identify which tracts to connect
plot_nyc_region_ids <- function(nyc_gdf) {
  nyc_gdf <- nyc_gdf %>%
    mutate(region_id = row_number())  # Explicitly store row indices
  nyc_coords = st_coordinates(st_centroid(nyc_gdf['geometry']))
  label_data <- data.frame(
    X = nyc_coords[,1],
    Y = nyc_coords[,2],
    region_id = nyc_gdf$region_id
  )
  p <- ggplot(nyc_gdf) +
      geom_sf(fill = "white", color = "black", size = 0.3) +
      geom_text(data = label_data, aes(x = X, y = Y, label = region_id), size=1) +
      theme_minimal()
  return(p)
}
