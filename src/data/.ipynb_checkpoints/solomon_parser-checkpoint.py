"""
Solomon Benchmark Dataset Parser
Parses Solomon VRP benchmark instances (C, R, RC types)
Supports both .txt and .csv formats
"""

import numpy as np
import pandas as pd
import csv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class Customer:
    """Customer data structure"""
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float


@dataclass
class VRPInstance:
    """VRP instance data structure"""
    name: str
    vehicle_capacity: float
    depot: Customer
    customers: List[Customer]
    type: str  # C, R, or RC
    num_customers: int
    
    def get_distance_matrix(self) -> np.ndarray:
        """Calculate Euclidean distance matrix"""
        n = len(self.customers) + 1  # +1 for depot
        coords = np.array([[self.depot.x, self.depot.y]] + 
                         [[c.x, c.y] for c in self.customers])
        
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt((coords[i][0] - coords[j][0])**2 + 
                                  (coords[i][1] - coords[j][1])**2)
                    dist_matrix[i][j] = dist
        
        return dist_matrix


class SolomonParser:
    """Parser for Solomon benchmark instances"""
    
    @staticmethod
    def parse(file_path: str, vehicle_capacity: Optional[float] = None) -> VRPInstance:
        """
        Parse Solomon benchmark file (supports .txt and .csv formats)
        
        Args:
            file_path: Path to Solomon instance file
            vehicle_capacity: Vehicle capacity (if None, will try to detect or use defaults)
            
        Returns:
            VRPInstance object
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Extract instance name and type
        instance_name = file_path.replace('\\', '/').split('/')[-1].split('.')[0]
        instance_type = instance_name[0] if instance_name else 'R'  # C, R, or RC
        
        # Determine capacity based on folder structure (C1/R1/RC1 = 200, C2/R2/RC2 = 1000)
        if vehicle_capacity is None:
            # Check parent folder name
            parent_folder = file_path.replace('\\', '/').split('/')[-2].upper() if '/' in file_path.replace('\\', '/') else ''
            if '1' in parent_folder or instance_name.endswith('01') or instance_name.endswith('02'):
                vehicle_capacity = 200.0  # C1, R1, RC1 series
            elif '2' in parent_folder:
                vehicle_capacity = 1000.0  # C2, R2, RC2 series
            else:
                vehicle_capacity = 200.0  # Default
        
        # Parse based on file extension
        if file_ext == '.csv':
            return SolomonParser._parse_csv(file_path, instance_name, instance_type, vehicle_capacity)
        else:
            return SolomonParser._parse_txt(file_path, instance_name, instance_type, vehicle_capacity)
    
    @staticmethod
    def _parse_csv(file_path: str, instance_name: str, instance_type: str, 
                   vehicle_capacity: float) -> VRPInstance:
        """Parse CSV format Solomon instance"""
        df = pd.read_csv(file_path)
        
        # Debug: Print original columns
        original_columns = list(df.columns)
        
        # Normalize column names: strip whitespace, convert to uppercase, handle variations
        # Map common variations to expected column names
        column_mapping = {}
        expected_columns = {
            'CUST NO.': ['CUST NO.', 'CUST_NO', 'CUSTNO', 'CUSTOMER NO', 'CUSTOMER_NO', 'ID'],
            'XCOORD.': ['XCOORD.', 'XCOORD', 'X', 'X_COORD', 'X_COORDINATE'],
            'YCOORD.': ['YCOORD.', 'YCOORD', 'Y', 'Y_COORD', 'Y_COORDINATE'],
            'DEMAND': ['DEMAND', 'DEM'],
            'READY TIME': ['READY TIME', 'READY_TIME', 'READYTIME', 'READY', 'READY_TIME_WINDOW'],
            'DUE DATE': ['DUE DATE', 'DUE_DATE', 'DUEDATE', 'DUE', 'DUE_TIME', 'DUE_TIME_WINDOW'],
            'SERVICE TIME': ['SERVICE TIME', 'SERVICE_TIME', 'SERVICETIME', 'SERVICE', 'SERVICE_TIME_WINDOW']
        }
        
        # Create normalized column mapping (handle whitespace and case)
        df_columns_upper = {col.upper().strip(): col for col in df.columns}
        
        for expected_col, variations in expected_columns.items():
            for variation in variations:
                var_upper = variation.upper().strip()
                if var_upper in df_columns_upper:
                    column_mapping[df_columns_upper[var_upper]] = expected_col
                    break
        
        # If mapping is incomplete, try to match by partial name
        if len(column_mapping) < len(expected_columns):
            for col in df.columns:
                col_upper = col.upper().strip()
                if col not in column_mapping:
                    # Try partial matching
                    if 'CUST' in col_upper or (col_upper == 'ID' or col_upper.startswith('ID')):
                        if 'CUST NO.' not in column_mapping.values():
                            column_mapping[col] = 'CUST NO.'
                    elif 'XCOORD' in col_upper or (col_upper.startswith('X') and 'COORD' in col_upper) or col_upper == 'X':
                        if 'XCOORD.' not in column_mapping.values():
                            column_mapping[col] = 'XCOORD.'
                    elif 'YCOORD' in col_upper or (col_upper.startswith('Y') and 'COORD' in col_upper) or col_upper == 'Y':
                        if 'YCOORD.' not in column_mapping.values():
                            column_mapping[col] = 'YCOORD.'
                    elif 'DEMAND' in col_upper:
                        if 'DEMAND' not in column_mapping.values():
                            column_mapping[col] = 'DEMAND'
                    elif 'READY' in col_upper and 'TIME' in col_upper:
                        if 'READY TIME' not in column_mapping.values():
                            column_mapping[col] = 'READY TIME'
                    elif 'DUE' in col_upper and ('DATE' in col_upper or 'TIME' in col_upper):
                        if 'DUE DATE' not in column_mapping.values():
                            column_mapping[col] = 'DUE DATE'
                    elif 'SERVICE' in col_upper and 'TIME' in col_upper:
                        if 'SERVICE TIME' not in column_mapping.values():
                            column_mapping[col] = 'SERVICE TIME'
        
        # Rename columns to expected names
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Verify required columns exist after rename
        required_cols = ['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = (
                f"Error parsing CSV file: {file_path}\n"
                f"Missing required columns: {missing_cols}\n"
                f"Original columns in file: {original_columns}\n"
                f"Columns after mapping: {list(df.columns)}\n"
                f"Column mapping used: {column_mapping}\n"
                f"Please check if the CSV file has the correct format."
            )
            raise ValueError(error_msg)
        
        # Double-check: ensure all required columns are accessible
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Critical error: Column '{col}' not found after mapping in file {file_path}\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Original columns: {original_columns}"
                )
        
        # Clean data: strip whitespace and handle malformed values
        # Some CSV files may have extra whitespace or multiple values in cells
        for col in required_cols:
            if col in df.columns:
                # Convert to string first, then strip and clean
                df[col] = df[col].astype(str).str.strip()
                # Handle cases where cell contains multiple values (take first value)
                # Example: '0.00    1' -> '0.00'
                df[col] = df[col].str.split().str[0]
                # Replace empty strings with NaN
                df[col] = df[col].replace('', np.nan)
        
        # Helper functions for safe conversion
        def safe_float(value, default=0.0):
            """Safely convert value to float, handling whitespace and invalid values"""
            if pd.isna(value):
                return default
            if isinstance(value, (int, float)):
                return float(value)
            # Convert to string, strip, and take first value if multiple
            str_val = str(value).strip().split()[0] if str(value).strip() else str(default)
            try:
                return float(str_val)
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            """Safely convert value to int, handling whitespace and invalid values"""
            if pd.isna(value):
                return default
            if isinstance(value, (int, float)):
                return int(value)
            # Convert to string, strip, and take first value if multiple
            str_val = str(value).strip().split()[0] if str(value).strip() else str(default)
            try:
                return int(float(str_val))  # Convert via float first to handle "1.0"
            except (ValueError, TypeError):
                return default
        
        # Find depot (id=0 or first row with demand=0)
        try:
            depot_row = df[df['DEMAND'] == 0].iloc[0] if len(df[df['DEMAND'] == 0]) > 0 else df.iloc[0]
        except (IndexError, KeyError) as e:
            raise ValueError(f"Error finding depot in CSV file {file_path}: {e}. Available columns: {list(df.columns)}")
        
        try:
            depot = Customer(
                id=0,
                x=safe_float(depot_row['XCOORD.']),
                y=safe_float(depot_row['YCOORD.']),
                demand=0.0,
                ready_time=safe_float(depot_row['READY TIME']),
                due_time=safe_float(depot_row['DUE DATE']),
                service_time=safe_float(depot_row['SERVICE TIME'])
            )
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Error parsing depot row in CSV file {file_path}: {e}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Depot row columns: {list(depot_row.index) if hasattr(depot_row, 'index') else 'N/A'}\n"
                f"Column mapping: {column_mapping}"
            )
        
        # Parse customers (exclude depot)
        customers = []
        depot_cust_no = safe_int(depot_row['CUST NO.'])
        
        try:
            # Iterate using iloc for reliable column access
            for idx in range(len(df)):
                try:
                    # Access row using iloc for reliable column access
                    row = df.iloc[idx]
                    
                    # Verify all required columns are accessible
                    for col in required_cols:
                        if col not in row.index:
                            raise KeyError(f"Column '{col}' not found in row index")
                    
                    # Get values using safe conversion
                    cust_no_val = safe_int(row['CUST NO.'])
                    demand_val = safe_float(row['DEMAND'])
                    
                    # Skip depot (demand=0 and same customer number as depot)
                    if demand_val == 0 and cust_no_val == int(depot_cust_no):
                        continue
                    
                    customer = Customer(
                        id=cust_no_val,
                        x=safe_float(row['XCOORD.']),
                        y=safe_float(row['YCOORD.']),
                        demand=demand_val,
                        ready_time=safe_float(row['READY TIME']),
                        due_time=safe_float(row['DUE DATE']),
                        service_time=safe_float(row['SERVICE TIME'])
                    )
                    customers.append(customer)
                except (KeyError, ValueError, IndexError) as e:
                    # Show actual row values for debugging
                    row_values = {col: str(row[col]) if col in row.index else 'MISSING' for col in required_cols}
                    raise ValueError(
                        f"Error parsing customer row {idx} in CSV file {file_path}: {e}\n"
                        f"Expected columns: {required_cols}\n"
                        f"Row values: {row_values}\n"
                        f"Available columns in dataframe: {list(df.columns)}\n"
                        f"Dataframe shape: {df.shape}"
                    )
        except Exception as e:
            raise ValueError(
                f"Error iterating through customers in CSV file {file_path}: {e}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Column mapping: {column_mapping}"
            )
        
        # Sort by customer ID
        customers.sort(key=lambda c: c.id)
        
        return VRPInstance(
            name=instance_name,
            vehicle_capacity=vehicle_capacity,
            depot=depot,
            customers=customers,
            type=instance_type,
            num_customers=len(customers)
        )
    
    @staticmethod
    def _parse_txt(file_path: str, instance_name: str, instance_type: str, 
                   vehicle_capacity: float) -> VRPInstance:
        """Parse TXT format Solomon instance"""
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Try to find vehicle capacity in file (if not already set)
        found_capacity = vehicle_capacity
        depot_line_idx = -1
        
        for i, line in enumerate(lines):
            parts = line.upper().split()
            if 'CAPACITY' in parts or 'VEHICLE' in parts:
                # Try to extract capacity value
                for part in parts:
                    try:
                        found_capacity = float(part)
                        break
                    except ValueError:
                        continue
            # Look for depot (usually has id 0 or appears after header)
            if len(line.split()) >= 7:
                try:
                    data = [float(x) for x in line.split()]
                    if int(data[0]) == 0 or (depot_line_idx == -1 and i > 4):
                        depot_line_idx = i
                except (ValueError, IndexError):
                    continue
        
        # If capacity not found, try line 4 (common format)
        if found_capacity == 0.0 and len(lines) > 4:
            try:
                parts = lines[4].split()
                found_capacity = float(parts[1]) if len(parts) > 1 else vehicle_capacity
            except (ValueError, IndexError):
                found_capacity = vehicle_capacity
        
        # Use found capacity or fall back to provided
        final_capacity = found_capacity if found_capacity > 0 else vehicle_capacity
        
        # Find depot
        depot = None
        if depot_line_idx >= 0:
            try:
                depot_data = [float(x) for x in lines[depot_line_idx].split()]
                depot = Customer(
                    id=0,
                    x=depot_data[1],
                    y=depot_data[2],
                    demand=0.0,
                    ready_time=depot_data[4] if len(depot_data) > 4 else 0.0,
                    due_time=depot_data[5] if len(depot_data) > 5 else 1000.0,
                    service_time=depot_data[6] if len(depot_data) > 6 else 0.0
                )
            except (ValueError, IndexError):
                pass
        
        # If depot not found, use first customer line or create default
        if depot is None:
            # Try to find first valid customer line
            for i in range(max(5, depot_line_idx + 1), len(lines)):
                try:
                    data = [float(x) for x in lines[i].split()]
                    if len(data) >= 3:
                        depot = Customer(
                            id=0,
                            x=data[1] if len(data) > 1 else 40.0,
                            y=data[2] if len(data) > 2 else 50.0,
                            demand=0.0,
                            ready_time=0.0,
                            due_time=1000.0,
                            service_time=0.0
                        )
                        depot_line_idx = i
                        break
                except (ValueError, IndexError):
                    continue
        
        if depot is None:
            # Default depot
            depot = Customer(0, 40.0, 50.0, 0.0, 0.0, 1000.0, 0.0)
        
        # Parse customers (skip header and depot)
        customers = []
        start_idx = max(depot_line_idx + 1, 9)
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if not line or line.startswith('EOF'):
                break
            
            try:
                data = [float(x) for x in line.split()]
                if len(data) >= 7:
                    customer = Customer(
                        id=int(data[0]),
                        x=data[1],
                        y=data[2],
                        demand=data[3],
                        ready_time=data[4],
                        due_time=data[5],
                        service_time=data[6]
                    )
                    # Skip depot if it appears again
                    if customer.id != 0:
                        customers.append(customer)
            except (ValueError, IndexError):
                continue
        
        # Sort by customer ID
        customers.sort(key=lambda c: c.id)
        
        return VRPInstance(
            name=instance_name,
            vehicle_capacity=vehicle_capacity,
            depot=depot,
            customers=customers,
            type=instance_type,
            num_customers=len(customers)
        )
    
    @staticmethod
    def calculate_route_cost(route: List[int], instance: VRPInstance, 
                            dist_matrix: np.ndarray) -> float:
        """
        Calculate total cost/distance of a route
        
        Args:
            route: List of customer IDs (0 is depot)
            instance: VRP instance
            dist_matrix: Pre-computed distance matrix
            
        Returns:
            Total route cost
        """
        if len(route) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(route) - 1):
            total_cost += dist_matrix[route[i]][route[i+1]]
        
        return total_cost
    
    @staticmethod
    def validate_solution(routes: List[List[int]], instance: VRPInstance) -> Tuple[bool, str]:
        """
        Validate VRP solution
        
        Args:
            routes: List of routes (each route is list of customer IDs)
            instance: VRP instance
            
        Returns:
            (is_valid, error_message)
        """
        # Check all customers are visited exactly once
        visited = set()
        for route in routes:
            for customer_id in route:
                if customer_id == 0:  # Skip depot
                    continue
                if customer_id in visited:
                    return False, f"Customer {customer_id} visited multiple times"
                visited.add(customer_id)
        
        # Check all customers are visited
        expected_customers = set(range(1, instance.num_customers + 1))
        if visited != expected_customers:
            missing = expected_customers - visited
            return False, f"Missing customers: {missing}"
        
        # Check capacity constraints
        for route_idx, route in enumerate(routes):
            total_demand = sum(instance.customers[c-1].demand 
                             for c in route if c != 0)
            if total_demand > instance.vehicle_capacity:
                return False, f"Route {route_idx} exceeds capacity: {total_demand} > {instance.vehicle_capacity}"
        
        return True, "Valid solution"

