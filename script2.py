import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time

class GameOfLife:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        # Grid now stores color values: 0=dead, 1=red, 2=blue, 3=green, 4=yellow
        self.grid = np.zeros((height, width), dtype=int)
        self.running = False
        self.color_names = {0: "dead", 1: "red", 2: "blue", 3: "green", 4: "yellow"}
        
    def set_grid(self, new_grid):
        """Set the grid to a new configuration"""
        self.grid = new_grid.copy()
        
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
        if sum(color_counts.values()) == 0:
            return 1  # Default to red if no neighbors
        return max(color_counts.keys(), key=lambda k: color_counts[k])
    
    def update(self):
        """Apply modified Conway's Game of Life rules with colors"""
        new_grid = np.zeros_like(self.grid)
        
        for y in range(self.height):
            for x in range(self.width):
                color_counts, total_neighbors, neighbors = self.get_neighbors_by_color(x, y)
                current_color = self.grid[y, x]
                
                if current_color > 0:  # Living cell
                    # Standard Conway rules
                    if total_neighbors in [2, 3]:
                        new_grid[y, x] = current_color  # Survives with same color
                    # NEW RULE: Color-based survival
                    elif total_neighbors >= 4:
                        # Check if at least 3 neighbors are the same color
                        max_color_count = max(color_counts.values())
                        if max_color_count >= 3:
                            # Survives and potentially changes to dominant color
                            dominant_color = self.get_dominant_color(color_counts)
                            new_grid[y, x] = dominant_color
                        # Otherwise dies (new_grid[y, x] remains 0)
                    # Otherwise dies (standard rules)
                
                else:  # Dead cell
                    # Standard birth rule: exactly 3 neighbors
                    if total_neighbors == 3:
                        # New cell takes the dominant color of its neighbors
                        dominant_color = self.get_dominant_color(color_counts)
                        new_grid[y, x] = dominant_color
        
        self.grid = new_grid
        
    def clear(self):
        """Clear the grid"""
        self.grid = np.zeros((self.height, self.width), dtype=int)

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
    def __init__(self, width=80, height=60):
        self.game = GameOfLife(width, height)
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
        
        # Enable zoom and pan with matplotlib's built-in navigation
        # This gives us zoom (rectangle tool) and pan automatically
        
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
        
        print("Initialized with test pattern - you should see a small square in the center")
        print("Left-click and drag to paint alive cells")
        print("Right-click and drag to erase cells")
        print("Press 1,2,3,4 to select colors: Red, Blue, Green, Yellow")
        print("Middle-click to toggle between paint/erase modes")
        print("Scroll wheel to change brush size")
        print("Use toolbar: Zoom (rectangle), Pan (hand), Home (reset view)")
        print("Keyboard shortcuts: + (faster), - (slower), SPACE (pause/resume)")
        print("Arrow keys: pan around the grid")
        print("NEW RULES: Cells with 4+ neighbors survive if 3+ neighbors are same color!")

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
        """Handle mouse scroll for brush size"""
        if event.inaxes == self.ax:
            # Check if we're in zoom mode (Ctrl+scroll) or brush mode
            if event.key == 'control':
                # Zoom functionality
                if event.button == 'up':
                    self.zoom_level *= 1.2
                elif event.button == 'down':
                    self.zoom_level /= 1.2
                self.update_view()
                print(f"Zoom level: {self.zoom_level:.2f}")
            else:
                # Brush size functionality
                if event.button == 'up':
                    self.brush_size = min(self.brush_size + 1, 5)
                elif event.button == 'down':
                    self.brush_size = max(self.brush_size - 1, 1)
                print(f"Brush size: {self.brush_size}")
    
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
            self.pan_x = 0
            self.pan_y = 0
            self.zoom_level = 1.0
            self.update_view()
            print("View reset to home")
    
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
        """Animation function"""
        if self.game.running:
            self.game.update()
            self.generation += 1
            self.im.set_array(self.game.grid)
            self.ax.set_title(f'Conway\'s Game of Life - Generation {self.generation}')
        return [self.im]
    
    def clear_grid(self, event):
        """Clear the grid"""
        self.game.clear()
        self.generation = 0
        self.im.set_array(self.game.grid)
        self.ax.set_title('Multi-Color Conway\'s Life - Generation 0')
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
        self.ax.set_title('Multi-Color Conway\'s Life - Mandelbrot Set Loaded')
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
        self.ax.set_title('Multi-Color Conway\'s Life - Julia Set Loaded')
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
        self.ax.set_title('Multi-Color Conway\'s Life - Random Pattern Loaded')
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
        self.ax.set_title('Multi-Color Conway\'s Life - Walker Attempt (Experimental)')
        plt.draw()
        self.fig.canvas.flush_events()
    
    def run(self):
        """Start the visualization"""
        # Skip tight_layout to avoid warnings with buttons
        # plt.tight_layout()  # Commented out to avoid button layout conflicts
        
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
    
    # Create and run the visualizer
    visualizer = GameOfLifeVisualizer(width=80, height=60)
    visualizer.run()
