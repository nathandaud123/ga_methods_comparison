"""
Route Visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from ..data.solomon_parser import VRPInstance, Customer


class RoutePlotter:
    """Visualize VRP routes"""
    
    def __init__(self, instance: VRPInstance):
        self.instance = instance
    
    def plot_routes(self, routes: List[List[int]], 
                    title: str = "VRP Solution",
                    save_path: str = None,
                    show: bool = True):
        """
        Plot VRP routes
        
        Args:
            routes: List of routes (each route is list of customer IDs)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot depot
        ax.scatter(self.instance.depot.x, self.instance.depot.y, 
                  c='red', s=200, marker='s', label='Depot', zorder=5)
        ax.text(self.instance.depot.x, self.instance.depot.y, 
               'Depot', fontsize=10, ha='center', va='bottom')
        
        # Color palette for routes
        colors = plt.cm.tab20(np.linspace(0, 1, len(routes)))
        
        # Plot each route
        for route_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            
            color = colors[route_idx]
            
            # Plot customers in route
            route_x = [self.instance.depot.x]
            route_y = [self.instance.depot.y]
            
            for customer_id in route[1:-1]:  # Skip depot at start/end
                customer = self.instance.customers[customer_id - 1]
                route_x.append(customer.x)
                route_y.append(customer.y)
                ax.scatter(customer.x, customer.y, c=[color], s=100, zorder=3)
                ax.text(customer.x, customer.y, str(customer_id), 
                       fontsize=8, ha='center', va='bottom')
            
            route_x.append(self.instance.depot.x)
            route_y.append(self.instance.depot.y)
            
            # Draw route lines
            ax.plot(route_x, route_y, color=color, linewidth=2, 
                   alpha=0.6, label=f'Route {route_idx + 1}')
        
        ax.set_xlabel('X Coordinate', fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontweight='bold')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Route plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

