import pandas as pd
import io

def format_as_markdown(df, title):
    """
    Manually formats a Pandas DataFrame as a Markdown table
    to avoid needing the 'tabulate' library.
    """
    print(f"### {title}")
    
    # Create the header row
    headers = df.columns
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    print(header_row)
    print(separator_row)
    
    # Create the data rows
    for _, row in df.iterrows():
        # Format numbers to 2 decimal places
        formatted_values = [f"{x:.2f}" if isinstance(x, (int, float)) else str(x) for x in row]
        print("| " + " | ".join(formatted_values) + " |")
    
    print("\n")

def main():
    try:
        # Load the results
        csv_file = "logs/results.csv"
        df = pd.read_csv(csv_file)
        
        # Clean up: Force numeric types and drop invalid rows
        df['improvement_percent'] = pd.to_numeric(df['improvement_percent'], errors='coerce')
        df['time_taken_sec'] = pd.to_numeric(df['time_taken_sec'], errors='coerce')
        df = df.dropna(subset=['improvement_percent'])

        print("--- COPY BELOW THIS LINE ---\n")

        # 1. By Problem Size (Num Cities)
        group_cities = df.groupby('num_cities')[['improvement_percent', 'time_taken_sec']].mean()
        group_cities.columns = ['Avg Improvement (%)', 'Avg Time (s)']
        group_cities = group_cities.reset_index() # Make 'num_cities' a column again for printing
        format_as_markdown(group_cities, "Impact of Problem Size ($N$)")

        # 2. By Beta (The most critical parameter)
        group_beta = df.groupby('beta')[['improvement_percent', 'time_taken_sec']].mean()
        group_beta.columns = ['Avg Improvement (%)', 'Avg Time (s)']
        group_beta = group_beta.reset_index()
        format_as_markdown(group_beta, "Impact of Cost Exponent ($\\beta$)")

        # 3. By Alpha
        group_alpha = df.groupby('alpha')[['improvement_percent', 'time_taken_sec']].mean()
        group_alpha.columns = ['Avg Improvement (%)', 'Avg Time (s)']
        group_alpha = group_alpha.reset_index()
        format_as_markdown(group_alpha, "Impact of Weight Factor ($\\alpha$)")
        
        # 4. By Density
        group_density = df.groupby('density')[['improvement_percent', 'time_taken_sec']].mean()
        group_density.columns = ['Avg Improvement (%)', 'Avg Time (s)']
        group_density = group_density.reset_index()
        format_as_markdown(group_density, "Impact of Graph Density")

    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Make sure you are in the project root.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()