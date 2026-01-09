import numpy as np

def transform(input_grid):
    """Transform the input grid according to the pattern observed in ARC examples."""
    grid = np.array(input_grid)

    # Define the four corner 2x2 blocks
    # Top-left: rows 0-1, cols 0-1
    # Top-right: rows 0-1, cols 3-4
    # Bottom-left: rows 3-4, cols 0-1
    # Bottom-right: rows 3-4, cols 3-4

    # For each corner block, we need to swap the two anti-diagonal elements
    # with their counterparts in the opposite corner block

    # Make a copy to work with
    result = grid.copy()

    # Swap anti-diagonal elements between opposite corners

    # Top-left anti-diagonal: (0,1) and (1,0)
    # Bottom-right anti-diagonal: (3,4) and (4,3)
    # Swap (0,1) with (4,3) and (1,0) with (3,4)

    # Store values
    temp1 = result[0, 1]
    temp2 = result[1, 0]

    # Perform swaps
    result[0, 1], result[4, 3] = result[4, 3], temp1
    result[1, 0], result[3, 4] = result[3, 4], temp2

    # Top-right anti-diagonal: (0,3) and (1,4)
    # Bottom-left anti-diagonal: (3,0) and (4,1)
    # Swap (0,3) with (4,1) and (1,4) with (3,0)

    temp3 = result[0, 3]
    temp4 = result[1, 4]

    result[0, 3], result[4, 1] = result[4, 1], temp3
    result[1, 4], result[3, 0] = result[3, 0], temp4

    return result.tolist()

# Test with provided examples
if __name__ == "__main__":
    # Test Example 1
    input_1 = [[7, 1, 7, 8, 0], [0, 8, 7, 7, 1], [7, 7, 7, 7, 7], [8, 7, 7, 7, 1], [0, 1, 7, 8, 5]]
    output_1 = [[5, 1, 7, 8, 0], [0, 8, 7, 5, 1], [7, 7, 7, 7, 7], [8, 5, 7, 0, 1], [0, 1, 7, 8, 5]]

    # Test Example 2
    input_2 = [[8, 9, 7, 9, 3], [3, 7, 7, 7, 8], [7, 7, 7, 7, 7], [8, 7, 7, 7, 8], [2, 9, 7, 9, 2]]
    output_2 = [[8, 9, 7, 9, 3], [3, 2, 7, 2, 8], [7, 7, 7, 7, 7], [8, 3, 7, 3, 8], [2, 9, 7, 9, 2]]

    # Test Example 3
    input_3 = [[7, 4, 7, 4, 5], [4, 3, 7, 7, 3], [7, 7, 7, 7, 7], [5, 7, 7, 4, 3], [3, 4, 7, 5, 7]]
    output_3 = [[5, 4, 7, 4, 5], [4, 3, 7, 4, 3], [7, 7, 7, 7, 7], [5, 4, 7, 4, 3], [3, 4, 7, 5, 4]]

    # Test the function
    print("Test 1:", transform(input_1) == output_1)
    print("Test 2:", transform(input_2) == output_2)
    print("Test 3:", transform(input_3) == output_3)

    # Test with provided test input
    test_input = [[8, 7, 7, 7, 8], [2, 4, 7, 4, 9], [7, 7, 7, 7, 7], [9, 7, 7, 7, 9], [4, 2, 7, 2, 8]]
    print("\nTest input transformation:")
    result = transform(test_input)
    for row in result:
        print(row) 