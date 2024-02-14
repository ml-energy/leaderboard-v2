import pandas as pd
import matplotlib.pyplot as plt

def printable(name):
    # printable names of servers
    servers = {
        'tgi': 'TGI',
        'vllm': 'vLLM',
    }
    return servers[name]

# Read the data from the CSV file
name = 'a40'
df = pd.read_csv(f'{name}.csv')

# Plot latency/throughput curve for each model and server
servers = df['server'].unique()
models = df['model'].unique()

for server in servers:
    plt.figure()  # Create a new figure for each server

    max_y = 0  # Variable to store the maximum y-axis value

    for model in models:
        model_data = df[(df['server'] == server) & (df['model'] == model)]
        plt.plot(model_data['throughput'], model_data['latency'], marker='x', label=model)

        # Annotate each cross with the request rate
        for i, txt in enumerate(model_data['request-rate']):
            plt.annotate(f'{txt}', (model_data['throughput'].iloc[i], model_data['latency'].iloc[i]))

        max_y = max(max_y, model_data['latency'].max())  # Update max_y

    plt.xlabel('Throughput (req/s)')
    plt.ylabel('Latency (s)')
    plt.title(f'{printable(server)}: Latency vs. Throughput')
    plt.legend()
    
    # Set x-axis and y-axis limits starting from 0.0
    plt.xlim(0.0, df['throughput'].max()*1.1)
    plt.ylim(0.0, 500)
    # plt.ylim(0.0, max_y*1.1)
    
    plt.savefig(f'{name}_{server}_latency_throughput.png')
    plt.close()

    # Plot energy/throughput curve for each model and server
    plt.figure()  # Create a new figure for each server

    max_y = 0  # Reset max_y for the second plot

    for model in models:
        model_data = df[(df['server'] == server) & (df['model'] == model)]
        plt.plot(model_data['throughput'], model_data['energy'], marker='x', label=model)

        # Annotate each cross with the request rate
        for i, txt in enumerate(model_data['request-rate']):
            plt.annotate(f'{txt}', (model_data['throughput'].iloc[i], model_data['energy'].iloc[i]))

        max_y = max(max_y, model_data['energy'].max())  # Update max_y

    plt.xlabel('Throughput (req/s)')
    plt.ylabel('Energy (J)')
    plt.title(f'{printable(server)}: Energy vs. Throughput')
    plt.legend()
    
    # Set x-axis and y-axis limits starting from 0.0
    plt.xlim(0.0, df['throughput'].max()*1.1)
    plt.ylim(0.0, 3300)
    # plt.ylim(0.0, max_y*1.1)
    
    plt.savefig(f'{name}_{server}_energy_throughput.png')
