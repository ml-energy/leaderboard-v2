import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv('benchmark.csv')

# Plot latency/throughput curve for each model and server
servers = df['server'].unique()
models = df['model'].unique()

for server in servers:
    plt.figure()  # Create a new figure for each server
    for model in models:
        model_data = df[(df['server'] == server) & (df['model'] == model)]
        plt.plot(model_data['throughput'], model_data['latency'], marker='x', label=model)

        # Annotate each cross with the request rate
        for i, txt in enumerate(model_data['request-rate']):
            plt.annotate(f'{txt}', (model_data['throughput'].iloc[i], model_data['latency'].iloc[i]))

    plt.xlabel('Throughput')
    plt.ylabel('Latency')
    plt.title(f'Latency vs. Throughput for Server {server}')
    plt.legend()
    plt.savefig(f'latency_throughput_{server}.png')
    plt.close()

    # Plot energy/throughput curve for each model and server
    plt.figure()  # Create a new figure for each server
    for model in models:
        model_data = df[(df['server'] == server) & (df['model'] == model)]
        plt.plot(model_data['throughput'], model_data['energy'], marker='x', label=model)

        # Annotate each cross with the request rate
        for i, txt in enumerate(model_data['request-rate']):
            plt.annotate(f'{txt}', (model_data['throughput'].iloc[i], model_data['energy'].iloc[i]))

    plt.xlabel('Throughput')
    plt.ylabel('Energy')
    plt.title(f'Energy vs. Throughput for Server {server}')
    plt.legend()
    plt.savefig(f'energy_throughput_{server}.png')
    plt.close()
