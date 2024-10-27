def split_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Save the first 1000 lines to 'first_1000.txt'
    with open('./data/gowalla/Gowalla_totalCheckins.txt', 'w') as file1:
        file1.writelines(lines[:23895])
    
    # Save lines 1001 to 2000 to 'lines_1001_to_2000.txt'
    with open('./data/gowalla/test-dataset.txt', 'w') as file2:
        file2.writelines(lines[23895:47538])

# Usage
split_file('./data/gowalla/original-Gowalla_totalCheckins.txt')
