library(sf)
library(ggplot2)
library(tidyverse)

# get shapefiles, synthesize census tract ids in study data
tracts = st_read("nycTracts10/nyct2010.shp")
tracts$BoroFips = ifelse(tracts$BoroCode==1, "061", ifelse(tracts$BoroCode==2, "005", ifelse(tracts$BoroCode==3, "047", ifelse(tracts$BoroCode==4, "081", ifelse(tracts$BoroCode==5, "085", "")))))
tracts$ct2010full = paste0("36",tracts$BoroFips,tracts$CT2010)

# get study data, subset shapefiles to data in study
nyc_study = read_csv("nyc_kid_data.csv", col_types = cols("ct2010full" = col_character(), .default = col_guess()))

pop_tracts  =  tracts$ct2010full %in% nyc_study$ct2010full
nyc_poptracts  =  tracts[pop_tracts,]
nyc_poptracts  =  nyc_poptracts[order(nyc_poptracts$ct2010full),]


table(nyc_poptracts$ct2010full == nyc_study$ct2010full)["TRUE"] == dim(nyc_study)[1] # must be true


# add pop density per sq mi
#Shape_Area_sq_mi =  nyc_poptracts$Shape_Area / (5280 * 5280)
#pop_density =  nyc_study$pop0518 / Shape_Area_sq_mi
nyc_poptracts$kid_pop_per_sqmi = nyc_study$pop0518 / (nyc_poptracts$Shape_Area / (5280 * 5280))

# add predictors
nyc_poptracts$count = nyc_study$count
nyc_poptracts$kid_pop = nyc_study$pop0518
nyc_poptracts$pct_pubtransit = 1 - nyc_study$Pct_PrivVeh
nyc_poptracts$med_hh_inc = nyc_study$medhhinc1
nyc_poptracts$traffic = nyc_study$AADT1   ## traffic density, annual?
nyc_poptracts$frag_index = nyc_study$frag_index

# check that sf_dataframe map lines up with data

ggplot(data = nyc_poptracts) +
geom_sf(aes(fill = count)) +
scale_fill_gradient(low = "lightblue", high = "red") +
labs(title = "NYC School-Age Traffic Accident Victims by Census Tract")

ggplot(data = nyc_poptracts) +
geom_sf(aes(fill = kid_pop_density)) +
scale_fill_gradient(low = "lightblue", high = "red") +
    labs(title = "NYC Population Ages 5-18 Density by Census Tract",
         subtitle = "Density measured as population per square mile")

ggplot(data = nyc_poptracts) +
geom_sf(aes(fill = traffic)) +
scale_fill_gradient(low = "lightblue", high = "red") +
labs(title = "NYC Average Traffic by Census Tract")

ggplot(data = nyc_poptracts) +
geom_sf(aes(fill = med_hh_inc)) +
scale_fill_gradient(low = "lightblue", high = "red") +
labs(title = "NYC Median Household Income by Census Tract")

ggplot(data = nyc_poptracts) +
geom_sf(aes(fill = frag_index)) +
scale_fill_gradient(low = "lightblue", high = "red") +
labs(title = "NYC Fragmentation Index by Census Tract",
     subtitle = "Measures transiency, single households, other \"misery index\" indicators")


# save as GeoJSON

st_write(nyc_poptracts, "nyc_poptracts.geojson", driver = "GeoJSON")
