import pandas as pd
import numpy as np
import sys

# Check if the user provided a command line argument for prefill tokens
if len(sys.argv) > 2:
    try:
        prefill_token_value = int(sys.argv[1])
        decode_token_value = int(sys.argv[2])
    except ValueError:
        print("Please provide a valid integer for prefill tokens.")
        sys.exit(1)
else:
    prefill_token_value = 512  # Default value if no argument is provided
    decode_token_value = 32  # Default value for decode tokens

# Generate data based on the given requirements
num_prefill_tokens = [prefill_token_value] + [1] * 3  # 512 for the first request, 1 for the remaining requests
num_decode_tokens = [decode_token_value] * 4  # 32 decode tokens for each request

# Calculate total tokens and pd_ratio
num_total_tokens = [num_prefill + num_decode for num_prefill, num_decode in zip(num_prefill_tokens, num_decode_tokens)]
pd_ratio = [num_prefill / num_decode for num_prefill, num_decode in zip(num_prefill_tokens, num_decode_tokens)]

# Create a DataFrame
data = {
    'num_prefill_tokens': num_prefill_tokens,
    'num_decode_tokens': num_decode_tokens,
    'num_total_tokens': num_total_tokens,
    'pd_ratio': pd_ratio
}
df = pd.DataFrame(data)

# Save to CSV
csv_file_path = './data/processed_traces/padding_trace.csv'
df.to_csv(csv_file_path, index=False)
