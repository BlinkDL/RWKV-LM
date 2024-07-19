import re
import matplotlib.pyplot as plt
import sys
from datetime import datetime


#
# python3.10 src/plot-eval.py [input] [output]
#

# Function to extract floating point numbers from the text
def extract_numbers_from_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
        
    # Regular expression to match the pattern 'acc': XXX
    pattern = r"'acc': (\d+\.\d+)"
    matches = re.findall(pattern, data)
    
    # Convert matched strings to floating point numbers
    numbers = [float(match) for match in matches]
    
    return numbers

# Generate scatter plot and save to a file
def generate_scatter_plot(numbers, output_filename):
    plt.scatter(range(len(numbers)), numbers)
    plt.xlabel('Index')
    plt.ylabel('Accuracy')
    
    # Get the current date and time
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.title(f'Scatter Plot of Accuracies\n{current_datetime}')

    plt.savefig(output_filename)
    plt.close()

# Main function to read the file, extract numbers, and generate plot
def main():
    if len(sys.argv) != 3:
        filename = 'eval_log.txt'  # Replace with your file name
        output_filename = 'eval_log.png'  # Replace with your desired output file name
    else: 
        filename = sys.argv[1]  
        output_filename = sys.argv[2] 

    numbers = extract_numbers_from_file(filename)
    generate_scatter_plot(numbers, output_filename)

# Run the main function
if __name__ == "__main__":
    main()
