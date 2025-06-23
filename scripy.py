import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time

class GameOfLife:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.running = False
        
    def set_grid(self, new_grid):
        """Set the grid to a new configuration"""
        self.grid = new_grid.copy()
        
    def get_neighbors(self, x, y):
        """Count living neighbors around cell (x, y)"""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                count += self.grid[ny, nx]
        return count
    
    def update(self):
        """Apply Conway's Game of Life rules"""
        new_grid = np.zeros_like(self.grid)
        
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.get_neighbors(x, y)
                
                # Conway's rules:
                # 1. Live cell with 2-3 neighbors survives
                # 2. Dead cell with exactly 3 neighbors becomes alive
                # 3. All other cells die or stay dead
                if self.grid[y, x] == 1:  # Living cell
                    if neighbors in [2, 3]:
                        new_grid[y, x] = 1
                else:  # Dead cell
                    if neighbors == 3:
                        new_grid[y, x] = 1
        
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
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Initialize with a test pattern to verify display works
        test_grid = np.zeros((height, width))
        test_grid[height//2-2:height//2+3, width//2-2:width//2+3] = 1  # Small square in center
        
        self.im = self.ax.imshow(test_grid, cmap='RdYlBu', interpolation='nearest', vmin=0, vmax=1)
        self.ax.set_title('Conway\'s Game of Life - Click and drag to paint! Right-click to erase. Scroll to change brush size')
        self.ax.set_xlabel('Grid X')
        self.ax.set_ylabel('Grid Y')
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Add control buttons
        self.setup_buttons()
        
        # Animation control
        self.ani = None
        self.generation = 0
        
        print("Initialized with test pattern - you should see a small square in the center")
        print("Left-click and drag to paint alive cells (blue)")
        print("Right-click and drag to erase cells")
        print("Middle-click to toggle between paint/erase modes")
        print("Scroll wheel to change brush size")

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
            if event.button == 'up':
                self.brush_size = min(self.brush_size + 1, 5)
            elif event.button == 'down':
                self.brush_size = max(self.brush_size - 1, 1)
            print(f"Brush size: {self.brush_size}")
    
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
                        self.game.grid[grid_y, grid_x] = 1  # Paint alive
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
        
    def toggle_animation(self, event):
        """Start or stop the animation"""
        if self.ani is None:
            self.ani = animation.FuncAnimation(self.fig, self.animate, 
                                             interval=200, blit=False, cache_frame_data=False)
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
        self.ax.set_title('Conway\'s Game of Life - Generation 0')
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
        self.im.set_clim(vmin=0, vmax=1)  # Ensure proper color scaling
        self.ax.set_title('Conway\'s Game of Life - Mandelbrot Set Loaded')
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
        self.im.set_clim(vmin=0, vmax=1)
        self.ax.set_title('Conway\'s Game of Life - Julia Set Loaded')
        plt.draw()
        self.fig.canvas.flush_events()
    
    def load_random(self, event):
        """Load random pattern"""
        print("Loading random pattern...")
        random_grid = np.random.choice([0, 1], size=(self.game.height, self.game.width), p=[0.7, 0.3])
        print(f"Generated random grid with {np.sum(random_grid)} alive cells")
        self.game.set_grid(random_grid)
        self.generation = 0
        self.im.set_array(self.game.grid)
        self.im.set_clim(vmin=0, vmax=1)
        self.ax.set_title('Conway\'s Game of Life - Random Pattern Loaded')
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
