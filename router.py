import itertools

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import shapely.geometry as sg
import shapely.ops as so
from networkx import MultiGraph, NetworkXNoPath

from graph_types import Edge, NodeName, TypedMultiGraph


class Router:
    def __init__(self, heuristic="dijkstra", CRS=None, CRS_map=None):
        self.heuristic = heuristic
        self.CRS = CRS
        self.CRS_map = CRS_map

    def extract_shortest_path_edges(self, G:MultiGraph, node_route, heuristic_f):
        edge_route:list[Edge] = []
        for u, v in itertools.pairwise(node_route):
            # Get all edges between consecutive nodes
            edges = G.get_edge_data(u, v)

            # If multiple edges exist, choose the one with minimal weight
            min_edge = min(edges.items(),
                        key=lambda x: x[1].get(heuristic_f, 1))
            edge_route.append((u, v, min_edge[0]))

            # Convert to GeoDataFrame
        return edge_route

    def get_route(self, G:TypedMultiGraph, origin_node:NodeName, dest_node:NodeName, heuristic_f:str='my_weight'):
        try:
            route = nx.shortest_path(G.inner, origin_node, dest_node, weight=heuristic_f)
        except NetworkXNoPath:
            return None
        assert isinstance(route,list)
        # G_route = nx.subgraph(G, route).copy()
        # df_route = momepy.nx_to_gdf(G_route, lines=True, points=False)
        edge_route = self.extract_shortest_path_edges(G.inner, route, heuristic_f)
        return edge_route


    def set_o_d_coords_crs(self, G, origin_coords, dest_coords):

        df_coords = pd.DataFrame({"coordinates": ["origin", "destination"],
       "latitude": [origin_coords[0], dest_coords[0]],
       "longitude": [origin_coords[1], dest_coords[1]]})

        # Create geodataframe
        gdf_coords = gpd.GeoDataFrame(
            df_coords, geometry=gpd.points_from_xy(df_coords.longitude, df_coords.latitude), crs=self.CRS_map
        )
        gdf_coords = gdf_coords.to_crs(self.CRS)
        gdf_coords = gdf_coords[['coordinates', 'geometry']]

        # Get origin and destination location
        origin_point = gdf_coords.loc[gdf_coords['coordinates'] == 'origin', 'geometry'].values[0]
        dest_point = gdf_coords.loc[gdf_coords['coordinates'] == 'destination', 'geometry'].values[0]
        origin_node_loc = so.nearest_points(origin_point, sg.MultiPoint(list(G.nodes)))[1]
        dest_node_loc = so.nearest_points(dest_point, sg.MultiPoint(list(G.nodes)))[1]
        origin_node = (origin_node_loc.x, origin_node_loc.y)
        dest_node = (dest_node_loc.x, dest_node_loc.y)
        return origin_node, dest_node, origin_node_loc, dest_node_loc, gdf_coords

    def set_o_d_coords(self, G, gdf_coords):
        gdf_coords = gdf_coords[['coordinates', 'geometry']]

        # Get origin and destination location
        origin_point = gdf_coords.loc[gdf_coords['coordinates'] == 'origin', 'geometry'].values[0]
        dest_point = gdf_coords.loc[gdf_coords['coordinates'] == 'destination', 'geometry'].values[0]
        origin_node_loc = so.nearest_points(origin_point, sg.MultiPoint(list(G.nodes)))[1]
        dest_node_loc = so.nearest_points(dest_point, sg.MultiPoint(list(G.nodes)))[1]
        origin_node = (origin_node_loc.x, origin_node_loc.y)
        dest_node = (dest_node_loc.x, dest_node_loc.y)
        return origin_node, dest_node, origin_node_loc, dest_node_loc, gdf_coords



