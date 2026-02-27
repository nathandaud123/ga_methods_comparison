
import os
from typing import List, Tuple, Optional
import numpy as np
from .solomon_parser import VRPInstance, Customer

class UchoaParser:
    """Parser for Uchoa (CVRPLIB) benchmark instances (.vrp files)"""

    @staticmethod
    def parse(file_path: str) -> VRPInstance:
        """
        Parse .vrp file
        """
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        name = ""
        dimension = 0
        capacity = 0.0
        
        node_coords = {}
        demands = {}
        depot_id = 0

        section = ""
        
        # Parse header and sections
        iterator = iter(lines)
        for line in iterator:
            if ":" in line:
                parts = [p.strip() for p in line.split(":", 1)]
                key, val = parts[0], parts[1]
                if key == "NAME":
                    name = val
                elif key == "DIMENSION":
                    dimension = int(val)
                elif key == "CAPACITY":
                    capacity = float(val)
                elif key in ["NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"]:
                   # Sometimes headers are followed by data immediately or on next line? 
                   # Standard CVRPLIB usually has just header then data.
                   pass
            
            if line.startswith("NODE_COORD_SECTION"):
                section = "COORDS"
                continue
            elif line.startswith("DEMAND_SECTION"):
                section = "DEMANDS"
                continue
            elif line.startswith("DEPOT_SECTION"):
                section = "DEPOT"
                continue
            elif line.startswith("EOF"):
                break
            
            # Parse data
            if section == "COORDS":
                parts = line.split()
                if len(parts) >= 3:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    node_coords[nid] = (x, y)
            elif section == "DEMANDS":
                parts = line.split()
                if len(parts) >= 2:
                    nid = int(parts[0])
                    dem = float(parts[1])
                    demands[nid] = dem
            elif section == "DEPOT":
                val = int(line)
                if val != -1:
                    depot_id = val

        # Construct VRPInstance
        # Assume depot_id is the depot.
        # SolomonParser expects depot object and list of customers.
        # Depot usually has demand 0 in CVRPLIB?
        
        if depot_id not in node_coords:
            # Fallback for depot if not explicitly found in DEPOT_SECTION, usually 1
            depot_id = 1
            
        depot_coords = node_coords[depot_id]
        depot_demand = demands.get(depot_id, 0.0)
        
        # Create Depot Customer
        # Use large time windows for CVRP (essentially infinite/unconstrained)
        depot = Customer(
            id=0, # Remap depot to 0 for consistency with SolomonParser
            x=depot_coords[0],
            y=depot_coords[1],
            demand=depot_demand,
            ready_time=0.0,
            due_time=100000.0, # Large arbitrary value
            service_time=0.0
        )
        
        customers = []
        # Remap other nodes
        # Original IDs are 1..N. Depot is one of them.
        # We remap customers to 1..N-1
        
        # Sort nodes by ID to ensure deterministic order
        sorted_ids = sorted([nid for nid in node_coords.keys() if nid != depot_id])
        
        for idx, nid in enumerate(sorted_ids, start=1):
            coords = node_coords[nid]
            dem = demands.get(nid, 0.0)
            
            cust = Customer(
                id=idx, # Remapped ID
                x=coords[0],
                y=coords[1],
                demand=dem,
                ready_time=0.0,
                due_time=100000.0,
                service_time=0.0
            )
            customers.append(cust)
            
        return VRPInstance(
            name=name,
            vehicle_capacity=capacity,
            depot=depot,
            customers=customers,
            type="CVRP",
            num_customers=len(customers)
        )
