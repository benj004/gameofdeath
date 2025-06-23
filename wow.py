import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time

class GameOfLife:
    def __init__(self, width=100, height=100, max_width=500, max_height=500):
        self.initial_width = width
        self.initial_height = height
        self.max_width = max_width
        self.max_height = max_height
        self.width = width
        self.height = height
        # Grid now stores color values: 0=dead, 1=red, 2=blue, 3=green, 4=yellow
        self.grid = np.zeros((height, width), dtype=np.uint8)  # Use uint8 for memory efficiency
        self.running = False
        self.color_names = {0: "dead", 1: "red", 2: "blue", 3: "green", 4: "yellow"}
        
        # Optimization: track active region to avoid processing empty areas
        self.active_bounds = [0, width, 0, height]  # [min_x, max_x, min_y, max_y]
        self.bounds_margin = 5  # Margin around active area
        
        # Chaos mode settings
        self.chaos_mode = False
        self.chaos_probability = 0.3  # 30% chance to use custom rules instead of Conway
        self.rule_randomness = 0.1  # 10% chance for completely random outcomes
        
    def update_active_bounds(self):
        """Update the bounds of the active (non-zero) region for optimization"""
        if np.sum(self.grid) == 0:  # Empty grid
            center_x, center_y = self.width // 2, self.height // 2
            self.active_bounds = [center_x-1, center_x+1, center_y-1, center_y+1]
            return
            
        # Find bounding box of all alive cells
        alive_cells = np.where(self.grid > 0)
        if len(alive_cells[0]) == 0:
            return
            
        min_y, max_y = int(np.min(alive_cells[0])), int(np.max(alive_cells[0]))
        min_x, max_x = int(np.min(alive_cells[1])), int(np.max(alive_cells[1]))
        
        # Add margin for neighbor calculations
        self.active_bounds = [
            max(0, min_x - self.bounds_margin),
            min(self.width, max_x + self.bounds_margin + 1),
            max(0, min_y - self.bounds_margin), 
            min(self.height, max_y + self.bounds_margin + 1)
        ]
    
    def set_chaos_mode(self, enabled, chaos_prob=0.3, random_prob=0.1):
        """Enable/disable chaos mode with custom probabilities"""
        self.chaos_mode = enabled
        self.chaos_probability = chaos_prob
        self.rule_randomness = random_prob
        mode_text = "ON" if enabled else "OFF"
        print(f"Chaos mode: {mode_text}")
        if enabled:
            print(f"  - {chaos_prob*100:.0f}% chance to use custom rules")
            print(f"  - {random_prob*100:.0f}% chance for random outcomes")
    
    def apply_original_conway_rules(self, x, y, current_color, total_neighbors, dominant_color):
        """Apply original Conway's Game of Life rules"""
        if current_color > 0:  # Living cell
            if total_neighbors in [2, 3]:
                return current_color  # Survives
            else:
                return 0  # Dies
        else:  # Dead cell
            if total_neighbors == 3:
                return dominant_color if dominant_color > 0 else 1  # Born
            else:
                return 0  # Stays dead
    
    def apply_custom_color_rules(self, x, y, current_color, color_counts, total_neighbors, dominant_color):
        """Apply our custom color-based rules"""
        if current_color > 0:  # Living cell
            # NEW DEATH RULE: Random death chance for overcrowded cells
            if total_neighbors >= 5:
                # 60% chance to die regardless of other rules when overcrowded
                if np.random.random() < 0.6:
                    return 0  # Dies from overcrowding stress
            
            # Standard Conway rules
            if total_neighbors in [2, 3]:
                return current_color  # Survives with same color
            
            # Custom rule 1: Color-based survival with 4+ neighbors
            elif total_neighbors >= 4:
                max_color_count = max(color_counts.values()) if color_counts and any(color_counts.values()) else 0
                if max_color_count >= 3:
                    return dominant_color if dominant_color > 0 else current_color  # Survives and changes to dominant color
                else:
                    return 0  # Dies
            
            # Custom rule 2: Same-color survival with only 1 neighbor
            elif total_neighbors == 1:
                if color_counts.get(current_color, 0) == 1:
                    return current_color  # Survives because same color
                else:
                    return 0  # Dies
            else:
                return 0  # Dies
        else:  # Dead cell
            if total_neighbors == 3:
                return dominant_color if dominant_color > 0 else 1  # Born with dominant color
            else:
                return 0  # Stays dead
    
    def expand_grid_if_needed(self, target_width, target_height):
        """Expand grid to accommodate new size requirements"""
        new_width = min(max(self.width, target_width), self.max_width)
        new_height = min(max(self.height, target_height), self.max_height)
        
        if new_width != self.width or new_height != self.height:
            # Create new larger grid
            new_grid = np.zeros((new_height, new_width), dtype=np.uint8)
            
            # Calculate offset to center old grid in new grid
            offset_y = (new_height - self.height) // 2
            offset_x = (new_width - self.width) // 2
            
            # Copy old grid to center of new grid
            new_grid[offset_y:offset_y + self.height, offset_x:offset_x + self.width] = self.grid
            
            self.grid = new_grid
            self.width = new_width
            self.height = new_height
            
            # Update active bounds
            self.active_bounds[0] += offset_x
            self.active_bounds[1] += offset_x
            self.active_bounds[2] += offset_y
            self.active_bounds[3] += offset_y
            
            print(f"Grid expanded to {self.width}x{self.height}")
            return True
        return False
    
    def set_max_size(self, max_width, max_height):
        """Set maximum grid size"""
        self.max_width = max_width
        self.max_height = max_height
        print(f"Maximum grid size set to {max_width}x{max_height}")
        
    def set_grid(self, new_grid):
        """Set the grid to a new configuration"""
        self.grid = new_grid.astype(np.uint8)
        self.height, self.width = new_grid.shape
        self.update_active_bounds()
    
    def get_neighbors_by_color(self, x, y):
        """Count living neighbors around cell (x, y) by color"""
        color_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # red, blue, green, yellow
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                neighbor_color = self.grid[ny, nx]
                if neighbor_color > 0:  # If alive
                    color_counts[neighbor_color] += 1
                    neighbors.append(neighbor_color)
        
        total_neighbors = sum(color_counts.values())
        return color_counts, total_neighbors, neighbors
    
    def get_dominant_color(self, color_counts):
        """Get the most common color among neighbors"""
        if isinstance(color_counts, dict):
            if sum(color_counts.values()) == 0:
                return 1  # Default to red if no neighbors
            return max(color_counts.keys(), key=lambda k: color_counts[k])
        else:  # numpy array
            if np.sum(color_counts[1:]) == 0:
                return 1
            return np.argmax(color_counts[1:]) + 1
    
    def update(self):
        """Apply modified Conway's Game of Life rules with colors - OPTIMIZED WITH CHAOS"""
        # Update active bounds first
        self.update_active_bounds()
        
        # Only process the active region
        min_x, max_x, min_y, max_y = self.active_bounds
        
        # Create a smaller working grid for just the active area
        active_width = max_x - min_x
        active_height = max_y - min_y
        
        if active_width <= 0 or active_height <= 0:
            return
            
        new_grid = self.grid.copy()
        
        # Only update cells in the active region
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                color_counts, total_neighbors, neighbors = self.get_neighbors_by_color(x, y)
                current_color = self.grid[y, x]
                dominant_color = self.get_dominant_color(color_counts)
                
                # Determine which rule set to use
                if self.chaos_mode:
                    # Random chance for completely chaotic outcome
                    if np.random.random() < self.rule_randomness:
                        # Completely random outcome
                        if current_color > 0:
                            # Living cell: random chance to survive, die, or change color
                            rand_outcome = np.random.random()
                            if rand_outcome < 0.4:
                                new_grid[y, x] = 0  # Dies
                            elif rand_outcome < 0.7:
                                new_grid[y, x] = current_color  # Survives
                            else:
                                new_grid[y, x] = np.random.randint(1, 5)  # Random color change
                        else:
                            # Dead cell: small chance to spontaneously come alive
                            if np.random.random() < 0.1:
                                new_grid[y, x] = np.random.randint(1, 5)  # Random birth
                            else:
                                new_grid[y, x] = 0  # Stays dead
                        continue
                    
                    # Choose between Conway rules and custom rules randomly
                    if np.random.random() < self.chaos_probability:
                        # Use custom color rules
                        result = self.apply_custom_color_rules(
                            x, y, current_color, color_counts, total_neighbors, dominant_color
                        )
                        new_grid[y, x] = result
                    else:
                        # Use original Conway rules
                        result = self.apply_original_conway_rules(
                            x, y, current_color, total_neighbors, dominant_color
                        )
                        new_grid[y, x] = result
                else:
                    # Normal mode: always use custom rules
                    result = self.apply_custom_color_rules(
                        x, y, current_color, color_counts, total_neighbors, dominant_color
                    )
                    new_grid[y, x] = result
        
        self.grid = new_grid
        
    def clear(self):
        """Clear the grid"""
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        center_x, center_y = self.width // 2, self.height // 2
        self.active_bounds = [center_x-1, center_x+1, center_y-1, center_y+1]

class MandelbrotGenerator:
    @staticmethod
    def mandelbrot_iteration(c, max_iter=100):
        """Calculate Mandelbrot iterations for complex number c"""
        z = 0
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter
    
    @staticmethod
    def generate_mandelbrot_grid(width, height, x_min=-2.5, x_max=1.5, 
                               y_min=-2.0, y_max=2.0, max_iter=50, threshold=10):
        """Generate a binary grid based on Mandelbrot set"""
        grid = np.zeros((height, width), dtype=int)
        
        for y in range(height):
            for x in range(width):
                # Map pixel coordinates to complex plane
                real = x_min + (x / width) * (x_max - x_min)
                imag = y_min + (y / height) * (y_max - y_min)
                c = complex(real, imag)
                
                # Calculate Mandelbrot iterations
                iterations = MandelbrotGenerator.mandelbrot_iteration(c, max_iter)
                
                # Convert to binary: cells with low iteration counts are "alive"
                if iterations < threshold:
                    grid[y, x] = 1
                    
        return grid
    
    @staticmethod
    def generate_julia_set(width, height, c_real=-0.7, c_imag=0.27015, 
                          x_min=-2, x_max=2, y_min=-2, y_max=2, 
                          max_iter=50, threshold=10):
        """Generate a Julia set pattern"""
        grid = np.zeros((height, width), dtype=int)
        c = complex(c_real, c_imag)
        
        for y in range(height):
            for x in range(width):
                # Map pixel coordinates to complex plane
                real = x_min + (x / width) * (x_max - x_min)
                imag = y_min + (y / height) * (y_max - y_min)
                z = complex(real, imag)
                
                # Julia set iteration
                iterations = 0
                while abs(z) <= 2 and iterations < max_iter:
                    z = z*z + c
                    iterations += 1
                
                if iterations < threshold:
                    grid[y, x] = 1
                    
        return grid

class GameOfLifeVisualizer:
    def __init__(self, width=80, height=60, max_width=400, max_height=300):
        self.initial_width = width
        self.initial_height = height
        self.game = GameOfLife(width, height, max_width, max_height)
        self.mandelbrot_gen = MandelbrotGenerator()
        
        # Mouse interaction state
        self.mouse_pressed = False
        self.paint_mode = True  # True = paint alive cells, False = erase cells
        self.brush_size = 1  # Brush radius
        self.animation_speed = 200  # milliseconds between frames
        self.current_color = 1  # 1=red, 2=blue, 3=green, 4=yellow
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.view_width = width
        self.view_height = height
        
        # Performance tracking
        self.performance_mode = False
        self.skip_frames = 1
        self.frame_counter = 0
        
        # Chaos mode intensity levels
        self.chaos_intensity = 0  # 0=off, 1=low, 2=medium, 3=high
        self.chaos_levels = [
            (False, 0.0, 0.0),      # Off
            (True, 0.2, 0.05),      # Low: 20% custom rules, 5% random
            (True, 0.5, 0.1),       # Medium: 50% custom rules, 10% random
            (True, 0.8, 0.2),       # High: 80% custom rules, 20% random
        ]
        
        # Set up the plot with navigation toolbar
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Initialize with a test pattern to verify display works
        test_grid = np.zeros((height, width))
        # Create a colorful test pattern
        test_grid[height//2-2:height//2+3, width//2-2:width//2+3] = 1  # Red square
        test_grid[height//2, width//2] = 2  # Blue center
        test_grid[height//2-1, width//2-1] = 3  # Green corner
        test_grid[height//2+1, width//2+1] = 4  # Yellow corner
        
        # Create custom colormap: 0=black, 1=red, 2=blue, 3=green, 4=yellow
        from matplotlib.colors import ListedColormap
        colors = ['black', 'red', 'blue', 'green', 'yellow']
        self.cmap = ListedColormap(colors)
        
        self.im = self.ax.imshow(test_grid, cmap=self.cmap, interpolation='nearest', vmin=0, vmax=4)
        self.ax.set_title('Multi-Color Conway\'s Life - Press 1,2,3,4 to change colors!')
        self.ax.set_xlabel('Grid X')
        self.ax.set_ylabel('Grid Y')
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add control buttons
        self.setup_buttons()
        
        # Animation control
        self.ani = None
        self.generation = 0
        self.auto_expand = True  # Whether to auto-expand grid
        
        print("Initialized with test pattern - you should see a small square in the center")
        print("Left-click and drag to paint alive cells")
        print("Right-click and drag to erase cells")
        print("Press 1,2,3,4 to select colors: Red, Blue, Green, Yellow")
        print("Middle-click to toggle between paint/erase modes")
        print("Scroll wheel to change brush size")
        print("Ctrl+Scroll or Shift+Scroll to zoom in/out at mouse position")
        print("Z key: zoom in at center, X key: zoom out at center")
        print("Use toolbar: Zoom (rectangle), Pan (hand), Home (reset view)")
        print("Keyboard shortcuts: + (faster), - (slower), SPACE (pause/resume)")
        print("Arrow keys: pan around the grid")
        print("NEW RULES:")
        print("1. Cells with 4+ neighbors survive if 3+ neighbors are same color")
        print("2. Cells with only 1 SAME-COLOR neighbor survive (normally would die)")
        print(f"Grid can expand up to {max_width}x{max_height} - paint beyond edges to expand!")
        print("PERFORMANCE CONTROLS:")
        print("P key: Toggle performance mode (reduces display updates)")
        print("Q key: Show performance statistics")
        print("CHAOS CONTROLS:")
        print("C key: Toggle chaos mode ON/OFF")
        print("B key: Cycle chaos intensity (Low/Medium/High)")  # Changed from V to B
        print("N key: Quick toggle Normal/Chaos modes")
        print("NEW RULES:")
        print("1. Cells with 4+ neighbors survive if 3+ neighbors are same color")
        print("2. Cells with only 1 SAME-COLOR neighbor survive (normally would die)")
        print("3. NEW: Cells with 5+ neighbors have 60% chance to die from overcrowding!")

    def update_title(self):
        """Update window title with current mode and generation"""
        chaos_status = "OFF"
        if self.game.chaos_mode:
            intensity_names = ["OFF", "LOW", "MEDIUM", "HIGH"]
            chaos_status = f"ON ({intensity_names[self.chaos_intensity]})"
        
        color_name = self.game.color_names.get(self.current_color, "unknown")
        title = f"Multi-Color Conway's Life - Gen {self.generation} | Chaos: {chaos_status} | Color: {color_name.upper()}"
        self.ax.set_title(title)

    def on_mouse_press(self, event):
        """Handle mouse press events"""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.mouse_pressed = True
            
            # Determine paint mode based on mouse button
            if event.button == 1:  # Left click
                self.paint_mode = True
            elif event.button == 3:  # Right click
                self.paint_mode = False
            elif event.button == 2:  # Middle click
                self.paint_mode = not self.paint_mode
                mode_text = "PAINT" if self.paint_mode else "ERASE"
                print(f"Switched to {mode_text} mode")
                return
            
            # Paint/erase the clicked cell
            self.paint_cell(event.xdata, event.ydata)
    
    def on_mouse_release(self, event):
        """Handle mouse release events"""
        self.mouse_pressed = False
    
    def on_mouse_drag(self, event):
        """Handle mouse drag events"""
        if (self.mouse_pressed and event.inaxes == self.ax and 
            event.xdata is not None and event.ydata is not None):
            self.paint_cell(event.xdata, event.ydata)
    
    def on_scroll(self, event):
        """Handle mouse scroll for brush size and zoom"""
        if event.inaxes == self.ax:
            if event.key == 'control':
                # Ctrl+scroll for zoom
                zoom_factor = 1.2 if event.button == 'up' else 1/1.2
                self.zoom_at_point(event.xdata, event.ydata, zoom_factor)
            elif event.key == 'shift':
                # Shift+scroll for zoom (alternative)
                zoom_factor = 1.2 if event.button == 'up' else 1/1.2
                self.zoom_at_point(event.xdata, event.ydata, zoom_factor)
            else:
                # Regular scroll for brush size
                if event.button == 'up':
                    self.brush_size = min(self.brush_size + 1, 10)
                elif event.button == 'down':
                    self.brush_size = max(self.brush_size - 1, 1)
                print(f"Brush size: {self.brush_size}")
    
    def zoom_at_point(self, x, y, zoom_factor):
        """Zoom in/out at a specific point"""
        if x is None or y is None:
            return
            
        # Get current view limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate current view center and size
        x_center = (xlim[1] + xlim[0]) / 2
        y_center = (ylim[1] + ylim[0]) / 2
        x_size = xlim[1] - xlim[0]
        y_size = ylim[1] - ylim[0]
        
        # Calculate new view size
        new_x_size = x_size / zoom_factor
        new_y_size = y_size / zoom_factor
        
        # Adjust center to zoom towards mouse position
        x_shift = (x - x_center) * (1 - 1/zoom_factor)
        y_shift = (y - y_center) * (1 - 1/zoom_factor)
        new_x_center = x_center + x_shift
        new_y_center = y_center + y_shift
        
        # Set new limits
        self.ax.set_xlim(new_x_center - new_x_size/2, new_x_center + new_x_size/2)
        self.ax.set_ylim(new_y_center - new_y_size/2, new_y_center + new_y_size/2)
        
        self.fig.canvas.draw_idle()
        print(f"Zoom: {zoom_factor:.2f}x at ({x:.1f}, {y:.1f})")
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == '1':
            self.current_color = 1
            print("Selected color: RED")
        elif event.key == '2':
            self.current_color = 2
            print("Selected color: BLUE")
        elif event.key == '3':
            self.current_color = 3
            print("Selected color: GREEN")
        elif event.key == '4':
            self.current_color = 4
            print("Selected color: YELLOW")
        elif event.key == '+' or event.key == '=':
            # Speed up animation
            self.animation_speed = max(self.animation_speed - 50, 10)
            print(f"Animation speed: {self.animation_speed}ms")
            if self.ani:
                self.restart_animation()
        elif event.key == '-':
            # Slow down animation
            self.animation_speed = min(self.animation_speed + 50, 1000)
            print(f"Animation speed: {self.animation_speed}ms")
            if self.ani:
                self.restart_animation()
        elif event.key == ' ':
            # Space bar to pause/resume
            self.toggle_animation(None)
        elif event.key == 'up':
            self.pan_y -= 5
            self.update_view()
        elif event.key == 'down':
            self.pan_y += 5
            self.update_view()
        elif event.key == 'left':
            self.pan_x -= 5
            self.update_view()
        elif event.key == 'right':
            self.pan_x += 5
            self.update_view()
        elif event.key == 'home':
            # Reset view
            self.reset_view()
            print("View reset to home")
        elif event.key == 'z':
            # Z key for zoom in
            center_x = (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) / 2
            center_y = (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) / 2
            self.zoom_at_point(center_x, center_y, 1.5)
        elif event.key == 'x':
            # X key for zoom out
            center_x = (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) / 2
            center_y = (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) / 2
            self.zoom_at_point(center_x, center_y, 1/1.5)
        elif event.key == 'p':
            # Toggle performance mode
            self.performance_mode = not self.performance_mode
            self.skip_frames = 3 if self.performance_mode else 1
            mode_text = "ON" if self.performance_mode else "OFF"
            print(f"Performance mode: {mode_text}")
        elif event.key == 'q':
            # Show performance statistics
            alive_cells = np.sum(self.game.grid > 0)
            print(f"Performance stats:")
            print(f"  - Generation: {self.generation}")
            print(f"  - Grid size: {self.game.width}x{self.game.height}")
            print(f"  - Alive cells: {alive_cells}")
            print(f"  - Animation speed: {self.animation_speed}ms")
            print(f"  - Performance mode: {'ON' if self.performance_mode else 'OFF'}")
        elif event.key == 'c':
            # Toggle chaos mode
            self.game.chaos_mode = not self.game.chaos_mode
            if self.game.chaos_mode and self.chaos_intensity == 0:
                self.chaos_intensity = 1  # Set to low if turning on
            if not self.game.chaos_mode:
                self.chaos_intensity = 0
            self.apply_chaos_settings()
            print(f"Chaos mode: {'ON' if self.game.chaos_mode else 'OFF'}")
        elif event.key == 'b':
            # Cycle chaos intensity (changed from 'v' to avoid zoom conflict)
            self.chaos_intensity = (self.chaos_intensity + 1) % 4
            self.apply_chaos_settings()
            intensity_names = ["OFF", "LOW", "MEDIUM", "HIGH"]
            print(f"Chaos intensity: {intensity_names[self.chaos_intensity]}")
        elif event.key == 'n':
            # Quick toggle normal/chaos
            if self.game.chaos_mode:
                self.chaos_intensity = 0
                self.game.chaos_mode = False
            else:
                self.chaos_intensity = 2  # Medium chaos
                self.game.chaos_mode = True
            self.apply_chaos_settings()
            print(f"Quick toggle - Chaos: {'ON (MEDIUM)' if self.game.chaos_mode else 'OFF'}")
    
    def apply_chaos_settings(self):
        """Apply current chaos intensity settings"""
        enabled, chaos_prob, random_prob = self.chaos_levels[self.chaos_intensity]
        self.game.set_chaos_mode(enabled, chaos_prob, random_prob)
    
    def reset_view(self):
        """Reset view to show entire grid"""
        self.ax.set_xlim(0, self.game.width)
        self.ax.set_ylim(self.game.height, 0)  # Flip Y for proper image orientation
        self.fig.canvas.draw_idle()
    
    def update_view(self):
        """Update the view based on current pan and zoom"""
        # Calculate view bounds
        center_x = self.game.width // 2 + self.pan_x
        center_y = self.game.height // 2 + self.pan_y
        
        half_width = (self.view_width // 2) / self.zoom_level
        half_height = (self.view_height // 2) / self.zoom_level
        
        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_height
        y_max = center_y + half_height
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_max, y_min)  # Flip Y axis for proper image orientation
        self.fig.canvas.draw_idle()
    
    def restart_animation(self):
        """Restart animation with new speed"""
        if self.ani:
            self.ani.event_source.stop()
            self.ani = animation.FuncAnimation(self.fig, self.animate, 
                                             interval=self.animation_speed, 
                                             blit=False, cache_frame_data=False)
    
    def paint_cell(self, x, y):
        """Paint or erase a cell at the given coordinates"""
        # Convert matplotlib coordinates to grid coordinates
        center_x = int(round(x))
        center_y = int(round(y))
        
        # Check if we need to expand grid
        if self.auto_expand:
            margin = 10  # Expand with some margin
            needed_width = max(center_x + self.brush_size + margin, self.game.width)
            needed_height = max(center_y + self.brush_size + margin, self.game.height)
            
            if self.game.expand_grid_if_needed(needed_width, needed_height):
                # Update the display if grid expanded
                self.im.set_array(self.game.grid)
                self.im.set_extent([0, self.game.width, self.game.height, 0])
        
        # Paint in a brush pattern around the center
        for dy in range(-self.brush_size + 1, self.brush_size):
            for dx in range(-self.brush_size + 1, self.brush_size):
                grid_x = center_x + dx
                grid_y = center_y + dy
                
                # Check if within brush radius and grid bounds
                if (dx*dx + dy*dy < self.brush_size*self.brush_size and 
                    0 <= grid_x < self.game.width and 0 <= grid_y < self.game.height):
                    
                    # Set cell state based on paint mode
                    if self.paint_mode:
                        self.game.grid[grid_y, grid_x] = self.current_color  # Paint with current color
                    else:
                        self.game.grid[grid_y, grid_x] = 0  # Erase
        
        # Update display
        self.im.set_array(self.game.grid)
        self.fig.canvas.draw_idle()  # Efficient redraw
        
    def setup_buttons(self):
        """Setup control buttons"""
        # Button positions
        button_height = 0.04
        button_width = 0.1
        button_spacing = 0.02
        
        # Start/Stop button
        ax_start = plt.axes([0.1, 0.02, button_width, button_height])
        self.btn_start = Button(ax_start, 'Start/Stop')
        self.btn_start.on_clicked(self.toggle_animation)
        
        # Clear button
        ax_clear = plt.axes([0.1 + button_width + button_spacing, 0.02, button_width, button_height])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self.clear_grid)
        
        # Mandelbrot button
        ax_mandelbrot = plt.axes([0.1 + 2*(button_width + button_spacing), 0.02, button_width, button_height])
        self.btn_mandelbrot = Button(ax_mandelbrot, 'Mandelbrot')
        self.btn_mandelbrot.on_clicked(self.load_mandelbrot)
        
        # Julia set button
        ax_julia = plt.axes([0.1 + 3*(button_width + button_spacing), 0.02, button_width, button_height])
        self.btn_julia = Button(ax_julia, 'Julia Set')
        self.btn_julia.on_clicked(self.load_julia)
        
        # Random button
        ax_random = plt.axes([0.1 + 4*(button_width + button_spacing), 0.02, button_width, button_height])
        self.btn_random = Button(ax_random, 'Random')
        self.btn_random.on_clicked(self.load_random)
        
        # Walker attempt button
        ax_walker = plt.axes([0.1 + 5*(button_width + button_spacing), 0.02, button_width, button_height])
        self.btn_walker = Button(ax_walker, 'Walker')
        self.btn_walker.on_clicked(self.load_walker_attempt)
        
        # Speed control buttons
        ax_faster = plt.axes([0.1 + 6*(button_width + button_spacing), 0.02, button_width/2, button_height])
        self.btn_faster = Button(ax_faster, 'Faster')
        self.btn_faster.on_clicked(self.speed_up)
        
        ax_slower = plt.axes([0.1 + 6*(button_width + button_spacing) + button_width/2, 0.02, button_width/2, button_height])
        self.btn_slower = Button(ax_slower, 'Slower')
        self.btn_slower.on_clicked(self.speed_down)
    
    def speed_up(self, event):
        """Speed up the animation"""
        self.animation_speed = max(self.animation_speed - 50, 10)
        print(f"Animation speed: {self.animation_speed}ms")
        if self.ani:
            self.restart_animation()
    
    def speed_down(self, event):
        """Slow down the animation"""
        self.animation_speed = min(self.animation_speed + 50, 1000)
        print(f"Animation speed: {self.animation_speed}ms")
        if self.ani:
            self.restart_animation()
    
    def toggle_animation(self, event):
        """Start or stop the animation"""
        if self.ani is None:
            self.ani = animation.FuncAnimation(self.fig, self.animate, 
                                             interval=self.animation_speed, blit=False, cache_frame_data=False)
            self.game.running = True
        else:
            self.ani.event_source.stop()
            self.ani = None
            self.game.running = False
        plt.draw()
    
    def animate(self, frame):
        """Animation function - OPTIMIZED"""
        if self.game.running:
            self.game.update()
            self.generation += 1
            
            # Only update display every skip_frames iterations for performance
            self.frame_counter += 1
            if self.frame_counter >= self.skip_frames:
                self.im.set_array(self.game.grid)
                self.update_title()  # Update title with current mode
                self.frame_counter = 0
                return [self.im]
            else:
                # Return empty list to skip display update
                return []
        return [self.im]
    
    def clear_grid(self, event):
        """Clear the grid"""
        self.game.clear()
        self.generation = 0
        self.im.set_array(self.game.grid)
        self.update_title()
        plt.draw()
    
    def load_mandelbrot(self, event):
        """Load Mandelbrot set pattern"""
        print("Loading Mandelbrot set...")
        mandelbrot_grid = self.mandelbrot_gen.generate_mandelbrot_grid(
            self.game.width, self.game.height, 
            x_min=-2.5, x_max=1.0, y_min=-1.5, y_max=1.5,
            max_iter=30, threshold=8
        )
        print(f"Generated Mandelbrot grid with {np.sum(mandelbrot_grid)} alive cells")
        self.game.set_grid(mandelbrot_grid)
        self.generation = 0
        self.im.set_array(self.game.grid)
        self.im.set_clim(vmin=0, vmax=4)  # Update for 4 colors
        self.update_title()
        plt.draw()
        self.fig.canvas.flush_events()  # Force immediate update
    
    def load_julia(self, event):
        """Load Julia set pattern"""
        print("Loading Julia set...")
        julia_grid = self.mandelbrot_gen.generate_julia_set(
            self.game.width, self.game.height,
            c_real=-0.7, c_imag=0.27015,
            max_iter=30, threshold=8
        )
        print(f"Generated Julia grid with {np.sum(julia_grid)} alive cells")
        self.game.set_grid(julia_grid)
        self.generation = 0
        self.im.set_array(self.game.grid)
        self.im.set_clim(vmin=0, vmax=4)
        self.update_title()
        plt.draw()
        self.fig.canvas.flush_events()
    
    def load_random(self, event):
        """Load random colorful pattern"""
        print("Loading random colorful pattern...")
        random_grid = np.zeros((self.game.height, self.game.width))
        
        # Create random pattern with all 4 colors
        for y in range(self.game.height):
            for x in range(self.game.width):
                if np.random.random() < 0.3:  # 30% chance of being alive
                    random_grid[y, x] = np.random.randint(1, 5)  # Random color 1-4
        
        print(f"Generated random colorful grid with {np.sum(random_grid > 0)} alive cells")
        self.game.set_grid(random_grid)
        self.generation = 0
        self.im.set_array(self.game.grid)
        self.im.set_clim(vmin=0, vmax=4)
        self.update_title()
        plt.draw()
        self.fig.canvas.flush_events()
    
    def load_walker_attempt(self, event):
        """Attempt to create a walking-like pattern"""
        print("Loading walker attempt...")
        
        # Create a custom pattern that might look like a moving figure
        walker_grid = np.zeros((self.game.height, self.game.width))
        
        # Start position (left side of screen)
        start_x, start_y = 10, self.game.height // 2
        
        # Try to create a "humanoid" spaceship pattern
        # This is experimental and may not work as intended
        walker_pattern = [
            [0, 1, 0, 0, 0],  # "head"
            [1, 1, 1, 0, 0],  # "torso"
            [0, 1, 0, 0, 0],  # "torso"
            [1, 0, 1, 0, 0],  # "legs" spread
            [1, 0, 0, 1, 0],  # "feet"
        ]
        
        # Place the pattern
        for i, row in enumerate(walker_pattern):
            for j, cell in enumerate(row):
                if (start_y + i < self.game.height and 
                    start_x + j < self.game.width):
                    walker_grid[start_y + i, start_x + j] = cell
        
        # Add some glider guns or oscillators nearby for interaction
        # This might create interesting emergent behavior
        
        print(f"Generated walker attempt with {np.sum(walker_grid)} alive cells")
        print("Note: This is experimental - it may not actually 'walk'!")
        
        self.game.set_grid(walker_grid)
        self.generation = 0
        self.im.set_array(self.game.grid)
        self.im.set_clim(vmin=0, vmax=4)
        self.update_title()
        plt.draw()
        self.fig.canvas.flush_events()
    
    def run(self):
        """Start the visualization"""
        # Ensure interactive backend
        plt.ion()  # Turn on interactive mode
        plt.show(block=True)  # Block until window is closed

# Example usage and additional pattern generators
def create_glider_pattern():
    """Create a simple glider pattern"""
    pattern = np.zeros((10, 10))
    # Glider pattern
    glider = [[0, 1, 0],
              [0, 0, 1],
              [1, 1, 1]]
    
    for i in range(3):
        for j in range(3):
            pattern[i+1, j+1] = glider[i][j]
    
    return pattern

def create_oscillator_pattern():
    """Create a blinker oscillator pattern"""
    pattern = np.zeros((5, 5))
    # Blinker pattern
    pattern[2, 1:4] = 1
    return pattern

# Main execution
if __name__ == "__main__":
    # Check and set matplotlib backend
    import matplotlib
    print(f"Current matplotlib backend: {matplotlib.get_backend()}")
    
    # Try to use a GUI backend
    gui_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg']
    for backend in gui_backends:
        try:
            matplotlib.use(backend)
            print(f"Successfully set backend to: {backend}")
            break
        except:
            continue
    else:
        print("Warning: No GUI backend available. Try installing tkinter or PyQt5")
        print("For Ubuntu/Debian: sudo apt-get install python3-tk")
        print("For other systems: pip install PyQt5")
    
    print("Conway's Game of Life with Mandelbrot Set Generator")
    print("Controls:")
    print("- Start/Stop: Begin/pause the simulation")
    print("- Clear: Clear the grid")
    print("- Mandelbrot: Load Mandelbrot set pattern")
    print("- Julia Set: Load Julia set pattern")
    print("- Random: Load random pattern")
    print("\nClick on buttons to interact with the simulation!")
    
    # Create and run the visualizer with custom max size
    # You can change these numbers to set maximum grid size
    visualizer = GameOfLifeVisualizer(width=80, height=60, max_width=800, max_height=600)
    visualizer.run()
