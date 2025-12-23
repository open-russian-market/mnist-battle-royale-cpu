import matplotlib.pyplot as plt
import numpy as np

# Data from your benchmark results
methods_data = {
    # Boosting methods (red with different shapes)
    'XGBoost': {'time': 71.51, 'accuracy': 97.46, 'color': 'red', 'marker': 'o'},
    'LightGBM': {'time': 33.15, 'accuracy': 97.30, 'color': 'red', 'marker': 's'},
    # 'CatBoost': {'time': 839.37, 'accuracy': 97.03, 'color': 'red', 'marker': '^'},

    # Random Forest (orange with different shapes)
    'Random Forest': {'time': 6.83, 'accuracy': 96.80, 'color': 'orange', 'marker': 'o'},

    # CNN methods (yellow with different shapes)
    'PyTorch CNN': {'time': 58.18, 'accuracy': 98.58, 'color': 'yellow', 'marker': 'o'},
    'Keras CNN': {'time': 28.64, 'accuracy': 98.72, 'color': 'yellow', 'marker': 's'},

    # Linear methods (black with different shapes)
    'Linear SVM': {'time': 27.42, 'accuracy': 90.73, 'color': 'black', 'marker': 'o'},
    'Logistic Regression': {'time': 6.29, 'accuracy': 91.96, 'color': 'black', 'marker': 's'},
    'PyTorch Linear': {'time': 7.36, 'accuracy': 92.15, 'color': 'black', 'marker': '^'},
    'Keras Linear': {'time': 6.28, 'accuracy': 92.09, 'color': 'black', 'marker': 'v'},

    # MLP methods (grey with different shapes)
    'PyTorch MLP': {'time': 11.56, 'accuracy': 97.34, 'color': 'grey', 'marker': 'o'},
    'Keras MLP': {'time': 10.94, 'accuracy': 97.50, 'color': 'grey', 'marker': 's'},

    # LSTM methods (purple with different shapes)
    'PyTorch LSTM': {'time': 66.09, 'accuracy': 98.44, 'color': 'purple', 'marker': 'o'},
    'Keras LSTM': {'time': 85.68, 'accuracy': 98.79, 'color': 'purple', 'marker': 's'},
}

def create_performance_plot():
    """Create scatter plot of test accuracy vs training time for different ML methods."""

    plt.figure(figsize=(12, 8))

    # Plot each method
    for method_name, data in methods_data.items():
        plt.scatter(data['time'], data['accuracy'],
                   c=data['color'], marker=data['marker'],
                   s=100, alpha=0.8, edgecolors='black', linewidth=1,
                   label=method_name)

    # Customize the plot
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('MNIST Classification: Accuracy vs Training Time', fontsize=14, fontweight='bold')

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Set axis limits with some padding
    plt.xlim(0, max([data['time'] for data in methods_data.values()]) * 1.1)
    plt.ylim(85, 100)

    plt.tight_layout()
    plt.savefig('mnist_battle_royale_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'mnist_battle_royale_results.png'")

    # Show some statistics
    print("\nPerformance Summary:")
    print("=" * 50)

    # Best accuracy
    best_acc = max([data['accuracy'] for data in methods_data.values()])
    best_methods = [name for name, data in methods_data.items() if data['accuracy'] == best_acc]
    print(".2f")

    # Fastest training
    fastest_time = min([data['time'] for data in methods_data.values()])
    fastest_methods = [name for name, data in methods_data.items() if data['time'] == fastest_time]
    print(".2f")

    # Best accuracy under 60 seconds
    fast_methods = {name: data for name, data in methods_data.items() if data['time'] < 60}
    if fast_methods:
        best_fast_acc = max([data['accuracy'] for data in fast_methods.values()])
        best_fast_methods = [name for name, data in fast_methods.items() if data['accuracy'] == best_fast_acc]
        print(".2f")

if __name__ == '__main__':
    create_performance_plot()